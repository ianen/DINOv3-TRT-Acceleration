param(
    [string]$CodeDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$TensorRtRoot = $env:TENSORRT_ROOT,
    [string]$BuildDir = "build\cpp-trt-inspect-msvc",
    [string]$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
)

$ErrorActionPreference = "Stop"

if (-not $TensorRtRoot) {
    $TensorRtRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.13.2.6"
}

if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "Visual Studio Developer Command Prompt not found: $VsDevCmd"
}

if (-not (Test-Path -LiteralPath (Join-Path $TensorRtRoot "include\NvInfer.h"))) {
    throw "TensorRT include path not found under: $TensorRtRoot"
}

$buildPath = Join-Path $CodeDir $BuildDir
$command = @(
    "`"$VsDevCmd`" -arch=x64",
    "cd /d `"$CodeDir`"",
    "set `"TENSORRT_ROOT=$TensorRtRoot`"",
    "cmake -S cpp -B `"$buildPath`" -G Ninja -DCMAKE_CXX_COMPILER=cl -DDINOV3_TRT_CPP_ENABLE_TENSORRT=ON",
    "cmake --build `"$buildPath`"",
    "ctest --test-dir `"$buildPath`" --output-on-failure"
) -join " && "

cmd.exe /c $command
