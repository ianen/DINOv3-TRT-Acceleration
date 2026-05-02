param(
    [string]$Engine = "Artifacts\engines\dinov3_vitl16_4out.bf16.prefer.engine",
    [int]$BatchSize = 8,
    [int]$ImageSize = 224,
    [int]$NumStreams = 4,
    [int]$Threads = 16,
    [int]$Warmup = 30,
    [int]$Iters = 200,
    [string]$OutputDir = "Artifacts\reports\utilization"
)

# Capture G7 utilization snapshot at the V1.0.3 G1 production-best config
# (N=4 slots × 16 caller threads with --pinned, the 720 qps configuration).
# Couples nvidia-smi 100ms cadence to pool_benchmark wallclock so the
# utilization window matches the throughput run exactly.

$ErrorActionPreference = "Stop"
$CodeDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bench = Join-Path $CodeDir "build\cpp-trt-inspect-msvc\dinov3_trt_pool_benchmark.exe"
$resolvedEngine = Join-Path $CodeDir $Engine
$resolvedOut = Join-Path $CodeDir $OutputDir
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runId = "v103_g1_pool_r{0}_b{1}_n{2}_t{3}_{4}" -f `
    $ImageSize, $BatchSize, $NumStreams, $Threads, $timestamp
$runDir = Join-Path $resolvedOut $runId
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$timelineCsv = Join-Path $runDir "timeline.csv"
$benchmarkJson = Join-Path $runDir "bench.json"

$queryFields = "timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm"
$smiArgs = @("--query-gpu=$queryFields", "--format=csv,nounits", "-lms", "100")

Write-Host "[g1_util] run_id=$runId  N=$NumStreams t=$Threads pinned"
Write-Host "[g1_util] starting nvidia-smi -> $timelineCsv"
$smiProc = Start-Process -FilePath "nvidia-smi" `
    -ArgumentList $smiArgs `
    -RedirectStandardOutput $timelineCsv `
    -NoNewWindow -PassThru
Start-Sleep -Seconds 1
$startUtc = (Get-Date).ToUniversalTime()

$benchArgs = @(
    "--engine", $resolvedEngine,
    "--batch-size", $BatchSize,
    "--image-size", $ImageSize,
    "--num-streams", $NumStreams,
    "--threads", $Threads,
    "--warmup", $Warmup,
    "--iters", $Iters,
    "--no-graphs",
    "--pinned"
)
Write-Host "[g1_util] running pool_benchmark..."
& $bench @benchArgs > $benchmarkJson
$endUtc = (Get-Date).ToUniversalTime()

if (-not $smiProc.HasExited) {
    Stop-Process -Id $smiProc.Id -Force
    $smiProc.WaitForExit(5000) | Out-Null
}
Write-Host "[g1_util] benchmark wallclock $([math]::Round(($endUtc - $startUtc).TotalSeconds, 2))s"

$meta = [pscustomobject]@{
    run_id              = $runId
    duration_seconds    = [int]([math]::Round(($endUtc - $startUtc).TotalSeconds))
    interval_ms         = 100
    warmup_skip_seconds = 1
    start_utc           = $startUtc.ToString("o")
    end_utc             = $endUtc.ToString("o")
    timeline_csv        = $timelineCsv
    benchmark_json      = $benchmarkJson
    query_fields        = $queryFields
    config              = "G1 production: r=$ImageSize b=$BatchSize N=$NumStreams t=$Threads pinned"
}
$meta | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $runDir "meta.json") -Encoding utf8

# Aggregate utilization
$pythonExe = Join-Path $CodeDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) { $pythonExe = "python" }
$aggregator = Join-Path $CodeDir "scripts\aggregate_utilization.py"
$summaryJson = Join-Path $runDir "summary.json"
& $pythonExe $aggregator $runDir --summary-json $summaryJson

# Print bench json + summary side-by-side
Write-Host ""
Write-Host "=== Throughput (pool_benchmark) ==="
Get-Content $benchmarkJson
Write-Host ""
Write-Host "=== Utilization (G7) ==="
Get-Content $summaryJson
