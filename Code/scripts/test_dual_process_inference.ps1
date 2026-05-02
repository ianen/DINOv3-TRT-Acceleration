param(
    [string]$Engine = "Artifacts\engines\dinov3_vitl16_4out.bf16.prefer.engine",
    [int]$BatchSize = 1,
    [int]$Warmup = 10,
    [int]$Iters = 100
)

# V1.0.3 ADR-020 follow-up: test whether TWO independent processes
# running TensorRT inference concurrently succeed where in-process
# multi-context fails. If processes work but threads don't, the Myelin
# blocker is process-internal state, and the multi-process pool variant
# is a viable workaround.

$ErrorActionPreference = "Stop"
$CodeDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bench = Join-Path $CodeDir "build\cpp-trt-inspect-msvc\dinov3_trt_runtime_benchmark.exe"
$resolvedEngine = Join-Path $CodeDir $Engine

Write-Host "[dual_process_test] launching 2 concurrent benchmark processes..."

$jobA = Start-Job -ScriptBlock {
    param($b, $e, $bs, $w, $i)
    & $b $e $bs $w $i 2>&1
} -ArgumentList @($bench, $resolvedEngine, $BatchSize, $Warmup, $Iters)

$jobB = Start-Job -ScriptBlock {
    param($b, $e, $bs, $w, $i)
    & $b $e $bs $w $i 2>&1
} -ArgumentList @($bench, $resolvedEngine, $BatchSize, $Warmup, $Iters)

Wait-Job -Job $jobA, $jobB | Out-Null

$resA = Receive-Job -Job $jobA
$resB = Receive-Job -Job $jobB
Remove-Job -Job $jobA, $jobB

Write-Host "===== Process A output ====="
$resA | ForEach-Object { Write-Host $_ }

Write-Host "===== Process B output ====="
$resB | ForEach-Object { Write-Host $_ }

# Heuristic verdict: if either process reports a Myelin error string,
# multi-process didn't help. Otherwise it's a viable workaround.
$failPattern = "Myelin|enqueueV3.*failed|Error Code"
$aBad = ($resA -join "`n") -match $failPattern
$bBad = ($resB -join "`n") -match $failPattern

if ($aBad -or $bBad) {
    Write-Host "[dual_process_test] VERDICT: one or both processes hit TRT errors — multi-process does NOT bypass Myelin blocker"
    exit 1
}
Write-Host "[dual_process_test] VERDICT: both processes succeeded — multi-process IS a viable workaround for the Myelin in-process thread-safety blocker"
exit 0
