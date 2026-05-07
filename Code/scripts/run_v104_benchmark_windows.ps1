param(
    [string]$CodeDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Dataset = "Artifacts\datasets\good_r512",
    [int[]]$BatchSizes = @(1, 4, 8, 16),
    [string[]]$Precisions = @("fp32", "bf16"),  # 必选; 可选 + "fp16", "int8"
    [int]$Warmup = 10,
    [int]$Iters = 100,
    [string]$OutputDir = "Artifacts\reports\v104_runs",
    [switch]$SkipPython,
    [switch]$SkipCpp
)

# V1.0.4 ADR-026 + ADR-027 全 sweep — 4 batch × N precision × 2 language
#
# 默认 8 必选数据点 (4 batch × 2 必选精度 × 2 language = 16 行 JSON)
# 可选 -Precisions @("fp32","bf16","fp16","int8") = 32 行

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $CodeDir

$resolvedDataset = Join-Path $CodeDir $Dataset
if (-not (Test-Path -LiteralPath $resolvedDataset)) {
    throw "dataset not found: $resolvedDataset"
}

$outDir = Join-Path $CodeDir $OutputDir
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$enginesDir = Join-Path $CodeDir "Artifacts\engines"
function Resolve-EnginePath {
    param([string]$Precision)
    switch ($Precision) {
        "fp32" { return (Join-Path $enginesDir "dinov3_vitl16_4out.r512.fp32.engine") }
        "bf16" { return (Join-Path $enginesDir "dinov3_vitl16_4out.r512.bf16.prefer.engine") }
        "fp16" { return (Join-Path $enginesDir "dinov3_vitl16_4out.r512.fp16.engine") }
        "int8" { return (Join-Path $enginesDir "dinov3_vitl16_4out.r512.int8.smoothquant_a08.engine") }
        default { throw "unknown precision: $Precision" }
    }
}

$pythonExe = Join-Path $CodeDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) { $pythonExe = "python" }

$cppBench = Join-Path $CodeDir "build\cpp-trt-inspect-msvc\dinov3_trt_production_benchmark.exe"

$totalRuns = 0
foreach ($p in $Precisions) {
    foreach ($b in $BatchSizes) {
        $enginePath = Resolve-EnginePath -Precision $p
        if (-not (Test-Path -LiteralPath $enginePath)) {
            Write-Warning "engine missing for precision=$p; skip: $enginePath"
            continue
        }

        $tag = "r512_${p}_b${b}"

        if (-not $SkipPython) {
            $pyOut = Join-Path $outDir "${tag}_py.json"
            Write-Host "[run_v104] Python  $tag → $pyOut"
            & $pythonExe "scripts\production_benchmark.py" `
                --engine $enginePath `
                --dataset $resolvedDataset `
                --batch-size $b `
                --image-size 512 `
                --warmup $Warmup `
                --iters $Iters `
                --output $pyOut
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Python benchmark failed for $tag (exit $LASTEXITCODE)"
            } else {
                $totalRuns += 1
            }
        }

        if (-not $SkipCpp) {
            if (-not (Test-Path -LiteralPath $cppBench)) {
                Write-Warning "cpp benchmark exe not found: $cppBench (run build first)"
                continue
            }
            $cppOut = Join-Path $outDir "${tag}_cpp.json"
            Write-Host "[run_v104] C++     $tag → $cppOut"
            & $cppBench `
                --engine $enginePath `
                --dataset $resolvedDataset `
                --batch-size $b `
                --image-size 512 `
                --warmup $Warmup `
                --iters $Iters `
                --output $cppOut
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "C++ benchmark failed for $tag (exit $LASTEXITCODE)"
            } else {
                $totalRuns += 1
            }
        }
    }
}

Write-Host "[run_v104] complete: $totalRuns successful runs in $outDir"
