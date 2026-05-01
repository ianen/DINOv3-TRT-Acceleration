param(
    [string]$CodeDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$BuildDir = "build\cpp-trt-inspect-msvc",
    [string]$Fp16Engine = "Artifacts\engines\dinov3_vitl16_4out.random.fp16.engine",
    [string]$Fp32Engine = "Artifacts\engines\dinov3_vitl16_4out.random.fp32.engine",
    [string]$Fp16Report = "Artifacts\reports\cpp_runtime_smoke_random_fp16_b1.json",
    [string]$Fp32Report = "Artifacts\reports\cpp_runtime_smoke_random_fp32_b1.json",
    [int]$BatchSize = 1
)

$ErrorActionPreference = "Stop"

$runner = Join-Path $CodeDir (Join-Path $BuildDir "dinov3_trt_runtime_smoke.exe")
if (-not (Test-Path -LiteralPath $runner)) {
    throw "C++ TensorRT runtime smoke executable not found: $runner"
}

function Invoke-RuntimeSmoke {
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

    $json = & $runner $resolvedEngine $BatchSize
    Set-Content -Path $resolvedReport -Value $json -Encoding UTF8
    Get-Item -LiteralPath $resolvedReport | Select-Object FullName, Length
}

Invoke-RuntimeSmoke -EnginePath $Fp16Engine -ReportPath $Fp16Report
Invoke-RuntimeSmoke -EnginePath $Fp32Engine -ReportPath $Fp32Report
