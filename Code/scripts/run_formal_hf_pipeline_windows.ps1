param(
    [string]$CodeDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$ExportPython = "python",
    [string]$ToolPython = ".venv\Scripts\python.exe",
    [string]$WeightDir = "Artifacts\weights\dinov3-vitl16-pretrain-lvd1689m",
    [string]$OnnxPath = "Artifacts\onnx\dinov3_vitl16_4out.onnx",
    [string]$Fp16Engine = "Artifacts\engines\dinov3_vitl16_4out.fp16.engine",
    [string]$Fp32Engine = "Artifacts\engines\dinov3_vitl16_4out.fp32.engine",
    [string]$Bf16Engine = "Artifacts\engines\dinov3_vitl16_4out.bf16.prefer.engine",
    [string]$Fp16TimingCache = "Artifacts\engines\dinov3_vitl16_4out.timing.cache",
    [string]$Fp32TimingCache = "Artifacts\engines\dinov3_vitl16_4out.fp32.timing.cache",
    [string]$Bf16TimingCache = "Artifacts\engines\dinov3_vitl16_4out.bf16.prefer.timing.cache",
    [string]$Fp16Benchmark = "Artifacts\reports\trtexec_fp16_smoke.json",
    [string]$Fp32Benchmark = "Artifacts\reports\trtexec_fp32_smoke.json",
    [string]$Bf16Benchmark = "Artifacts\reports\trtexec_bf16_prefer_smoke.json",
    [string]$CompareReport = "Artifacts\reports\trt_fp32_vs_fp16_b1.json",
    [string]$Bf16CompareReport = "Artifacts\reports\compare_fp32_vs_bf16_prefer_b1.json",
    [string]$Batches = "1,8,32",
    [int]$BenchmarkDuration = 3,
    [switch]$SkipExport,
    [switch]$SkipBuild,
    [switch]$SkipBenchmark,
    [switch]$SkipCompare,
    [switch]$SkipBf16
)

$ErrorActionPreference = "Stop"

function Resolve-CodePath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path $CodeDir $Path
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Output "BEGIN $Name"
    $global:LASTEXITCODE = 0
    & $Command
    if ($global:LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $global:LASTEXITCODE"
    }
    Write-Output "END $Name"
}

$resolvedWeightDir = Resolve-CodePath -Path $WeightDir
$resolvedOnnxPath = Resolve-CodePath -Path $OnnxPath
$resolvedFp16Engine = Resolve-CodePath -Path $Fp16Engine
$resolvedFp32Engine = Resolve-CodePath -Path $Fp32Engine
$resolvedBf16Engine = Resolve-CodePath -Path $Bf16Engine
$resolvedFp16TimingCache = Resolve-CodePath -Path $Fp16TimingCache
$resolvedFp32TimingCache = Resolve-CodePath -Path $Fp32TimingCache
$resolvedBf16TimingCache = Resolve-CodePath -Path $Bf16TimingCache
$resolvedFp16Benchmark = Resolve-CodePath -Path $Fp16Benchmark
$resolvedFp32Benchmark = Resolve-CodePath -Path $Fp32Benchmark
$resolvedBf16Benchmark = Resolve-CodePath -Path $Bf16Benchmark
$resolvedCompareReport = Resolve-CodePath -Path $CompareReport
$resolvedBf16CompareReport = Resolve-CodePath -Path $Bf16CompareReport

Set-Location $CodeDir

Invoke-Step -Name "check-assets-before" -Command {
    & $ToolPython scripts\check_assets.py --require weights
}

if (-not $SkipExport) {
    Invoke-Step -Name "export-formal-onnx" -Command {
        & $ExportPython scripts\export_hf_dinov3_onnx.py `
            --model-path $resolvedWeightDir `
            --local-files-only `
            --output $resolvedOnnxPath `
            --validate-no-if
    }
}

Invoke-Step -Name "inspect-formal-onnx" -Command {
    & $ExportPython scripts\inspect_onnx.py $resolvedOnnxPath
}

if (-not $SkipBuild) {
    Invoke-Step -Name "build-fp16-engine" -Command {
        & $ToolPython scripts\build_engine_trtexec.py `
            --onnx $resolvedOnnxPath `
            --engine $resolvedFp16Engine `
            --precision fp16 `
            --timing-cache $resolvedFp16TimingCache
    }
    Invoke-Step -Name "build-fp32-engine" -Command {
        & $ToolPython scripts\build_engine_trtexec.py `
            --onnx $resolvedOnnxPath `
            --engine $resolvedFp32Engine `
            --precision fp32 `
            --timing-cache $resolvedFp32TimingCache
    }
    if (-not $SkipBf16) {
        Invoke-Step -Name "build-bf16-prefer-engine" -Command {
            & $ToolPython scripts\build_engine_trtexec.py `
                --onnx $resolvedOnnxPath `
                --engine $resolvedBf16Engine `
                --precision bf16 `
                --precision-constraints prefer `
                --layer-precision "*:bf16" `
                --timing-cache $resolvedBf16TimingCache
        }
    }
}

if (-not $SkipBenchmark) {
    Invoke-Step -Name "benchmark-fp16-engine" -Command {
        & $ToolPython scripts\benchmark_trtexec.py `
            --engine $resolvedFp16Engine `
            --output $resolvedFp16Benchmark `
            --batches $Batches `
            --duration $BenchmarkDuration
    }
    Invoke-Step -Name "benchmark-fp32-engine" -Command {
        & $ToolPython scripts\benchmark_trtexec.py `
            --engine $resolvedFp32Engine `
            --output $resolvedFp32Benchmark `
            --batches $Batches `
            --duration $BenchmarkDuration
    }
    if (-not $SkipBf16) {
        Invoke-Step -Name "benchmark-bf16-prefer-engine" -Command {
            & $ToolPython scripts\benchmark_trtexec.py `
                --engine $resolvedBf16Engine `
                --output $resolvedBf16Benchmark `
                --batches $Batches `
                --duration $BenchmarkDuration
        }
    }
}

if (-not $SkipCompare) {
    Invoke-Step -Name "compare-fp32-vs-fp16-b1" -Command {
        & $ToolPython scripts\compare_trt_engines.py `
            --reference-engine $resolvedFp32Engine `
            --candidate-engine $resolvedFp16Engine `
            --output $resolvedCompareReport `
            --batch-size 1 `
            --seed 20260429
    }
    if (-not $SkipBf16) {
        Invoke-Step -Name "compare-fp32-vs-bf16-prefer-b1" -Command {
            & $ToolPython scripts\compare_trt_engines.py `
                --reference-engine $resolvedFp32Engine `
                --candidate-engine $resolvedBf16Engine `
                --output $resolvedBf16CompareReport `
                --batch-size 1 `
                --seed 20260429
        }
    }
}

Invoke-Step -Name "check-assets-after" -Command {
    if ($SkipBf16) {
        & $ToolPython scripts\check_assets.py --require weights --require onnx --require fp16-engine --require fp32-engine --require reports
    } else {
        & $ToolPython scripts\check_assets.py --require weights --require onnx --require fp16-engine --require fp32-engine --require bf16-engine --require reports
    }
}
