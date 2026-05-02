param(
    [string]$TrtRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.13.2.6",
    [string]$OnnxPath = "Artifacts\onnx\dinov3_vitl16_4out.onnx",
    [string]$EnginePath = "Artifacts\engines\dinov3_vitl16_4out.r224.bf16.prefer.b16.engine"
)

# V1.0.3 G1 push: build a r224 BF16-prefer engine with max_batch=16
# (V1.0.1 baseline engine maxed at b=8). Empirical question: does the
# r224 launch-overhead-dominated regime benefit from b=16 by amortizing
# more transfer cost over a single inference? Memory budget ~52 MB
# activation, well within VRAM.

$ErrorActionPreference = "Stop"
$CodeDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$trtexec = Join-Path $TrtRoot "bin\trtexec.exe"
if (-not (Test-Path -LiteralPath $trtexec)) {
    throw "trtexec.exe not found: $trtexec"
}
$resolvedOnnx = Join-Path $CodeDir $OnnxPath
$resolvedEngine = Join-Path $CodeDir $EnginePath

if (-not (Test-Path -LiteralPath $resolvedOnnx)) {
    throw "ONNX not found: $resolvedOnnx"
}

$args = @(
    "--onnx=$resolvedOnnx",
    "--saveEngine=$resolvedEngine",
    "--bf16",
    "--precisionConstraints=prefer",
    "--layerPrecisions=*:bf16",
    "--minShapes=pixel_values:1x3x224x224",
    "--optShapes=pixel_values:8x3x224x224",
    "--maxShapes=pixel_values:16x3x224x224",
    "--useSpinWait",
    "--skipInference"
)

Write-Host "[build_r224_b16] launching trtexec..."
& $trtexec @args
$exit = $LASTEXITCODE
Write-Host "[build_r224_b16] trtexec exit code: $exit"

if (Test-Path -LiteralPath $resolvedEngine) {
    $size = (Get-Item -LiteralPath $resolvedEngine).Length
    Write-Host "[build_r224_b16] engine built, size = $([math]::Round($size / 1MB, 1)) MB"
}
else {
    throw "engine not produced at: $resolvedEngine"
}
