param(
    [string]$CodeDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$BuildDir = "build\cpp-trt-inspect-msvc",
    [string]$Fp16Engine = "Artifacts\engines\dinov3_vitl16_4out.random.fp16.engine",
    [string]$Fp32Engine = "Artifacts\engines\dinov3_vitl16_4out.random.fp32.engine",
    [string]$Fp16Report = "Artifacts\reports\cpp_runtime_benchmark_random_fp16.json",
    [string]$Fp32Report = "Artifacts\reports\cpp_runtime_benchmark_random_fp32.json",
    [string]$Batches = "1,8,32",
    [int]$WarmupIterations = 10,
    [int]$Iterations = 50
)

$ErrorActionPreference = "Stop"

$runner = Join-Path $CodeDir (Join-Path $BuildDir "dinov3_trt_runtime_benchmark.exe")
if (-not (Test-Path -LiteralPath $runner)) {
    throw "C++ TensorRT runtime benchmark executable not found: $runner"
}

function Convert-BatchList {
    param([string]$Value)

    $items = @()
    foreach ($part in $Value.Split(",")) {
        $trimmed = $part.Trim()
        if (-not $trimmed) {
            continue
        }
        $batch = [int]$trimmed
        if ($batch -lt 1) {
            throw "Batch sizes must be >= 1: $batch"
        }
        $items += $batch
    }
    if ($items.Count -eq 0) {
        throw "At least one batch size is required"
    }
    return $items
}

function Invoke-RuntimeBenchmark {
    param(
        [string]$EnginePath,
        [string]$ReportPath,
        [int[]]$BatchList,
        [string]$Precision
    )

    $resolvedEngine = Join-Path $CodeDir $EnginePath
    $resolvedReport = Join-Path $CodeDir $ReportPath
    $reportDir = Split-Path -Parent $resolvedReport

    if (-not (Test-Path -LiteralPath $resolvedEngine)) {
        throw "Engine not found: $resolvedEngine"
    }
    New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

    $results = @()
    foreach ($batch in $BatchList) {
        $json = & $runner $resolvedEngine $batch $WarmupIterations $Iterations
        $results += ($json | ConvertFrom-Json)
    }

    $report = [ordered]@{
        engine_path = $resolvedEngine
        precision = $Precision
        batches = $BatchList
        warmup_iterations = $WarmupIterations
        iterations = $Iterations
        results = $results
    }
    $report | ConvertTo-Json -Depth 8 | Set-Content -Path $resolvedReport -Encoding UTF8
    Get-Item -LiteralPath $resolvedReport | Select-Object FullName, Length
}

$batchList = Convert-BatchList -Value $Batches
Invoke-RuntimeBenchmark -EnginePath $Fp16Engine -ReportPath $Fp16Report -BatchList $batchList -Precision "fp16"
Invoke-RuntimeBenchmark -EnginePath $Fp32Engine -ReportPath $Fp32Report -BatchList $batchList -Precision "fp32"
