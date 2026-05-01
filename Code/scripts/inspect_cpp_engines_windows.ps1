param(
    [string]$CodeDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$BuildDir = "build\cpp-trt-inspect-msvc",
    [string]$Fp16Engine = "Artifacts\engines\dinov3_vitl16_4out.random.fp16.engine",
    [string]$Fp32Engine = "Artifacts\engines\dinov3_vitl16_4out.random.fp32.engine",
    [string]$Fp16Report = "Artifacts\reports\cpp_engine_inspect_random_fp16.json",
    [string]$Fp32Report = "Artifacts\reports\cpp_engine_inspect_random_fp32.json"
)

$ErrorActionPreference = "Stop"

$inspector = Join-Path $CodeDir (Join-Path $BuildDir "dinov3_trt_inspect_engine.exe")
if (-not (Test-Path -LiteralPath $inspector)) {
    throw "C++ TensorRT inspector not found: $inspector"
}

function Invoke-Inspector {
    param(
        [string]$EnginePath,
        [string]$ReportPath
    )

    $resolvedEngine = Join-Path $CodeDir $EnginePath
    $resolvedReport = Join-Path $CodeDir $ReportPath
    $reportDir = Split-Path -Parent $resolvedReport

    if (-not (Test-Path -LiteralPath $resolvedEngine)) {
        throw "Engine not found: $resolvedEngine"
    }
    New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

    $json = & $inspector $resolvedEngine
    Set-Content -Path $resolvedReport -Value $json -Encoding UTF8
    Get-Item -LiteralPath $resolvedReport | Select-Object FullName, Length
}

Invoke-Inspector -EnginePath $Fp16Engine -ReportPath $Fp16Report
Invoke-Inspector -EnginePath $Fp32Engine -ReportPath $Fp32Report
