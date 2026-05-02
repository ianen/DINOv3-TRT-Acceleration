param(
    [Parameter(Mandatory = $true)]
    [string]$Engine,

    [Parameter(Mandatory = $false)]
    [int]$BatchSize = 1,

    [Parameter(Mandatory = $false)]
    [int]$ImageSize = 224,

    [Parameter(Mandatory = $false)]
    [string]$NumStreams = "1,2,4",

    [Parameter(Mandatory = $false)]
    [int]$Warmup = 30,

    [Parameter(Mandatory = $false)]
    [int]$Iters = 200,

    [Parameter(Mandatory = $false)]
    [string]$Precision = "bf16-prefer",

    [Parameter(Mandatory = $false)]
    [string]$Framework = "py_multistream",

    [Parameter(Mandatory = $false)]
    [string]$RunIdPrefix = "v103",

    [Parameter(Mandatory = $false)]
    [string]$BenchmarkCsv = "Artifacts\reports\v103_throughput_matrix.csv",

    [Parameter(Mandatory = $false)]
    [string]$BenchmarkJsonDir = "Artifacts\reports\v103_runs",

    [Parameter(Mandatory = $false)]
    [string]$UtilizationDir = "Artifacts\reports\utilization",

    [Parameter(Mandatory = $false)]
    [int]$IntervalMs = 100,

    [Parameter(Mandatory = $false)]
    [int]$WarmupSkipSeconds = 1
)

# V1.0.3 ADR-015/019/020 共用 — runs benchmark_multi_stream.py while
# capturing G7 utilization metrics in parallel, then aggregates and appends
# one row to the V1.0.3 §10.3 benchmark CSV per (resolution, batch, n_streams).
#
# Layout:
#   Code/                                  <- $CodeDir (script lives in scripts/)
#     scripts/
#       benchmark_multi_stream.py          existing V1.0.2 ADR-015 driver
#       utilization_monitor.ps1            G7 nvidia-smi/ncu sampler
#       aggregate_utilization.py           G7 row aggregator
#     Artifacts/
#       reports/
#         v103_throughput_matrix.csv       V1.0.3 §10.3 main CSV
#         v103_runs/<run_id>_bench.json    raw benchmark output
#         utilization/<run_id>/timeline.csv
#                              meta.json
#                              ncu_metrics.txt   (optional)

$ErrorActionPreference = "Stop"

$CodeDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-AbsPath([string]$p) {
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return Join-Path $CodeDir $p
}

$resolvedEngine = Resolve-AbsPath $Engine
if (-not (Test-Path -LiteralPath $resolvedEngine)) {
    throw "Engine not found: $resolvedEngine"
}

$resolvedCsv = Resolve-AbsPath $BenchmarkCsv
$resolvedJsonDir = Resolve-AbsPath $BenchmarkJsonDir
$resolvedUtilDir = Resolve-AbsPath $UtilizationDir

New-Item -ItemType Directory -Force -Path $resolvedJsonDir | Out-Null
New-Item -ItemType Directory -Force -Path $resolvedUtilDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resolvedCsv) | Out-Null

$pythonExe = Join-Path $CodeDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
    Write-Warning "[v103_runner] .venv\Scripts\python.exe not found, falling back to system python"
}

$benchmarkScript = Join-Path $CodeDir "scripts\benchmark_multi_stream.py"
if (-not (Test-Path -LiteralPath $benchmarkScript)) {
    throw "benchmark_multi_stream.py not found: $benchmarkScript"
}
$monitorScript = Join-Path $CodeDir "scripts\utilization_monitor.ps1"
if (-not (Test-Path -LiteralPath $monitorScript)) {
    throw "utilization_monitor.ps1 not found: $monitorScript"
}
$aggregatorScript = Join-Path $CodeDir "scripts\aggregate_utilization.py"
if (-not (Test-Path -LiteralPath $aggregatorScript)) {
    throw "aggregate_utilization.py not found: $aggregatorScript"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runId = "{0}_{1}_r{2}_b{3}_n{4}_{5}" -f `
    $RunIdPrefix, $Precision, $ImageSize, $BatchSize, ($NumStreams -replace ",", "-"), $timestamp
$benchmarkJson = Join-Path $resolvedJsonDir ($runId + "_bench.json")

Write-Host "[v103_runner] run_id=$runId"
Write-Host "[v103_runner] engine=$resolvedEngine"
Write-Host "[v103_runner] config: r=$ImageSize b=$BatchSize n_streams=$NumStreams warmup=$Warmup iters=$Iters"
Write-Host "[v103_runner] benchmark JSON -> $benchmarkJson"
Write-Host "[v103_runner] utilization     -> $resolvedUtilDir\$runId"
Write-Host "[v103_runner] benchmark CSV   -> $resolvedCsv"

$maxStreams = ($NumStreams.Split(",") | ForEach-Object { [int]$_.Trim() } | Measure-Object -Maximum).Maximum

# Inline nvidia-smi management: start as our own background process, run
# benchmark in foreground, stop nvidia-smi when benchmark exits. This makes
# the utilization window match the benchmark wallclock exactly — fixing the
# "monitor outlives benchmark and idle samples dilute SM%" issue from the
# original Wait-Job approach.

$runDir = Join-Path $resolvedUtilDir $runId
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$timelineCsv = Join-Path $runDir "timeline.csv"
$metaJson = Join-Path $runDir "meta.json"

$queryFields = "timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm"
$smiArgs = @(
    "--query-gpu=$queryFields",
    "--format=csv,nounits",
    "-lms", "$IntervalMs"
)

Write-Host "[v103_runner] starting nvidia-smi -> $timelineCsv"
$smiProc = Start-Process -FilePath "nvidia-smi" `
    -ArgumentList $smiArgs `
    -RedirectStandardOutput $timelineCsv `
    -NoNewWindow `
    -PassThru

# Brief settle so nvidia-smi flushes header + first samples before benchmark first H2D.
Start-Sleep -Seconds 1
$startUtc = (Get-Date).ToUniversalTime()

# Run benchmark in foreground.
Write-Host "[v103_runner] launching benchmark_multi_stream.py..."
$benchmarkArgs = @(
    $benchmarkScript,
    "--engine", $resolvedEngine,
    "--batch-size", $BatchSize,
    "--image-size", $ImageSize,
    "--num-streams", $NumStreams,
    "--warmup", $Warmup,
    "--iters", $Iters,
    "--output", $benchmarkJson
)
$benchProc = Start-Process -FilePath $pythonExe -ArgumentList $benchmarkArgs -NoNewWindow -PassThru -Wait
$endUtc = (Get-Date).ToUniversalTime()
if ($benchProc.ExitCode -ne 0) {
    Write-Warning "[v103_runner] benchmark exit code $($benchProc.ExitCode); stopping monitor anyway"
}

# Stop nvidia-smi.
if (-not $smiProc.HasExited) {
    Stop-Process -Id $smiProc.Id -Force
    $smiProc.WaitForExit(5000) | Out-Null
}
Write-Host "[v103_runner] nvidia-smi stopped, benchmark wallclock $([math]::Round(($endUtc - $startUtc).TotalSeconds, 2))s"

# Write meta.json so aggregate_utilization.py picks up the right interval/skip.
$meta = [pscustomobject]@{
    run_id              = $runId
    duration_seconds    = [int]([math]::Round(($endUtc - $startUtc).TotalSeconds))
    interval_ms         = $IntervalMs
    warmup_skip_seconds = $WarmupSkipSeconds
    start_utc           = $startUtc.ToString("o")
    end_utc             = $endUtc.ToString("o")
    timeline_csv        = $timelineCsv
    ncu_metrics_txt     = $null
    query_fields        = $queryFields
    benchmark_json      = $benchmarkJson
    benchmark_exit_code = $benchProc.ExitCode
}
$meta | ConvertTo-Json -Depth 4 | Set-Content -Path $metaJson -Encoding utf8

# Aggregate utilization -> append row to V1.0.3 §10.3 CSV.
$runDir = Join-Path $resolvedUtilDir $runId
$summaryJson = Join-Path $runDir "summary.json"

# Read benchmark JSON to extract aggregate_qps + p50/p99 (we append the row
# for the highest n_streams entry; the aggregator currently emits one
# utilization row per monitor run, not per stream count).
$aggregateQps = ""
$p50Latency = ""
$p99Latency = ""
$nInstances = $maxStreams
if (Test-Path -LiteralPath $benchmarkJson) {
    try {
        $benchData = Get-Content -LiteralPath $benchmarkJson -Raw | ConvertFrom-Json
        # benchmark_multi_stream.py schema: {"results": [{"n_streams", "aggregate_qps",
        # "median_latency_ms_p50_across_workers", "per_worker": [{"max_latency_ms"}], ...}]}
        if ($benchData.results) {
            $maxStreamRun = $benchData.results | Sort-Object -Property n_streams -Descending | Select-Object -First 1
            if ($maxStreamRun) {
                if ($maxStreamRun.PSObject.Properties["aggregate_qps"]) {
                    $aggregateQps = "{0:F2}" -f $maxStreamRun.aggregate_qps
                }
                if ($maxStreamRun.PSObject.Properties["median_latency_ms_p50_across_workers"]) {
                    $p50Latency = "{0:F3}" -f $maxStreamRun.median_latency_ms_p50_across_workers
                }
                # No p99 field; approximate with worker max_latency_ms (worst worker tail).
                if ($maxStreamRun.PSObject.Properties["per_worker"]) {
                    $maxLat = ($maxStreamRun.per_worker | Measure-Object -Property max_latency_ms -Maximum).Maximum
                    if ($maxLat) { $p99Latency = "{0:F3}" -f $maxLat }
                }
                if ($maxStreamRun.PSObject.Properties["n_streams"]) {
                    $nInstances = [int]$maxStreamRun.n_streams
                }
            }
        }
    }
    catch {
        Write-Warning "[v103_runner] could not parse benchmark JSON: $($_.Exception.Message)"
    }
}

$aggArgs = @(
    $aggregatorScript,
    $runDir,
    "--benchmark-csv", $resolvedCsv,
    "--summary-json", $summaryJson,
    "--row-field", ("source=" + (hostname)),
    "--row-field", ("framework=" + $Framework),
    "--row-field", ("precision=" + $Precision),
    "--row-field", ("resolution=" + $ImageSize),
    "--row-field", ("batch_size=" + $BatchSize),
    "--row-field", ("n_instances=" + $nInstances),
    "--row-field", "n_clients=",
    "--row-field", "preferred_batch_size=",
    "--row-field", "max_queue_delay_us="
)
if ($aggregateQps) { $aggArgs += @("--row-field", ("aggregate_qps=" + $aggregateQps)) }
if ($p50Latency) { $aggArgs += @("--row-field", ("p50_latency_ms=" + $p50Latency)) }
if ($p99Latency) { $aggArgs += @("--row-field", ("p99_latency_ms=" + $p99Latency)) }

Write-Host "[v103_runner] aggregating utilization -> CSV..."
& $pythonExe @aggArgs

Write-Host "[v103_runner] done. summary: $summaryJson"
Write-Host "[v103_runner] benchmark CSV row appended: $resolvedCsv"

# Output run_id so callers can chain.
Write-Output $runId
