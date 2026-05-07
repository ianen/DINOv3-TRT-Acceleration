param(
    [string]$CodeDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [int]$ImageSize = 512,
    [string]$OnnxPath = "Artifacts\onnx\dinov3_vitl16_4out.r512.onnx",
    [int]$MinBatch = 1,
    [int]$OptBatch = 4,
    [int]$MaxBatch = 16,
    [switch]$BuildFp16,
    [switch]$SkipFp32,
    [switch]$SkipBf16
)

# V1.0.4 ADR-025 — Build r=512 engines (FP32 + BF16 prefer 必选;
# 可选 FP16 via -BuildFp16)
#
# 前置条件:
#   - export_official_dinov3_onnx.py --image-size 512 已生成
#     Artifacts/onnx/dinov3_vitl16_4out.r512.onnx (~1.2 GB)
#   - 否则脚本先运行 export

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $CodeDir

$resolvedOnnx = Join-Path $CodeDir $OnnxPath

if (-not (Test-Path -LiteralPath $resolvedOnnx)) {
    Write-Host "[build_v104] ONNX not found, exporting..."
    & ".venv\Scripts\python.exe" "scripts\export_official_dinov3_onnx.py" `
        --image-size $ImageSize `
        --output $resolvedOnnx `
        --batch-size $OptBatch `
        --opset 18 `
        --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "ONNX export failed (exit $LASTEXITCODE)"
    }
}

$enginesDir = Join-Path $CodeDir "Artifacts\engines"
New-Item -ItemType Directory -Force -Path $enginesDir | Out-Null

function Build-Engine {
    param(
        [string]$Precision,
        [string]$EngineName,
        [string[]]$ExtraFlags
    )
    $enginePath = Join-Path $enginesDir $EngineName
    Write-Host "[build_v104] === $Precision engine ==="
    Write-Host "[build_v104] target: $enginePath"

    $args = @(
        "scripts\build_engine_trtexec.py",
        "--onnx", $resolvedOnnx,
        "--engine-out", $enginePath,
        "--image-size", $ImageSize,
        "--precision", $Precision,
        "--min-batch", $MinBatch,
        "--opt-batch", $OptBatch,
        "--max-batch", $MaxBatch,
        "--use-spin-wait"
    ) + $ExtraFlags

    & ".venv\Scripts\python.exe" $args
    if ($LASTEXITCODE -ne 0) {
        throw "$Precision engine build failed (exit $LASTEXITCODE)"
    }
    if (-not (Test-Path -LiteralPath $enginePath)) {
        throw "$Precision engine not produced at $enginePath"
    }
    $sz = (Get-Item -LiteralPath $enginePath).Length / 1MB
    Write-Host "[build_v104] $Precision engine built: $([math]::Round($sz, 1)) MB"
}

if (-not $SkipFp32) {
    Build-Engine -Precision "fp32" -EngineName "dinov3_vitl16_4out.r512.fp32.engine" -ExtraFlags @()
}

if (-not $SkipBf16) {
    Build-Engine -Precision "bf16" -EngineName "dinov3_vitl16_4out.r512.bf16.prefer.engine" `
        -ExtraFlags @("--precision-constraints", "prefer", "--layer-precisions", "*:bf16")
}

if ($BuildFp16) {
    Build-Engine -Precision "fp16" -EngineName "dinov3_vitl16_4out.r512.fp16.engine" -ExtraFlags @()
}

Write-Host "[build_v104] all engines built successfully"
