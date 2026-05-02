param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [Parameter(Mandatory = $false)]
    [int]$DurationSeconds = 30,

    [Parameter(Mandatory = $false)]
    [int]$IntervalMs = 100,

    [Parameter(Mandatory = $false)]
    [string]$RunId = "",

    [Parameter(Mandatory = $false)]
    [string]$NcuTargetExe = "",

    [Parameter(Mandatory = $false)]
    [string]$NcuTargetArgs = "",

    [Parameter(Mandatory = $false)]
    [int]$WarmupSkipSeconds = 5
)

# G7 GPU utilization monitor — V1.0.3 §10.1 deliverable.
# Spawns `nvidia-smi --query-gpu` at IntervalMs cadence for DurationSeconds.
# Optionally invokes Nsight Compute (ncu) for single-point per-kernel sampling.
# Output schema matches V1.0.3 §10.3 utilization timeline CSV.

$ErrorActionPreference = "Stop"

if (-not $RunId) {
    $RunId = (Get-Date -Format "yyyyMMdd-HHmmss")
}

$resolvedOut = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path (Get-Location) $OutputDir
}
$runDir = Join-Path $resolvedOut $RunId
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$timelineCsv = Join-Path $runDir "timeline.csv"
$ncuOut = Join-Path $runDir "ncu_metrics.txt"
$metaJson = Join-Path $runDir "meta.json"

Write-Host "[utilization_monitor] run_id=$RunId duration=${DurationSeconds}s interval=${IntervalMs}ms"
Write-Host "[utilization_monitor] output_dir=$runDir"

# Validate nvidia-smi available.
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    throw "nvidia-smi not found in PATH. Install NVIDIA driver / CUDA toolkit first."
}

# Compose nvidia-smi args.
# --query-gpu fields chosen for V1.0.3 G7 acceptance (utilization.gpu = SM%, utilization.memory = HBM controller %, memory.used, power.draw, temperature.gpu, clocks.sm).
$queryFields = "timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm"
$smiArgs = @(
    "--query-gpu=$queryFields",
    "--format=csv,nounits",
    "-lms", "$IntervalMs"
)

Write-Host "[utilization_monitor] launching nvidia-smi -> $timelineCsv"

# Start nvidia-smi background process with stdout redirected to CSV.
$smiProc = Start-Process -FilePath "nvidia-smi" `
    -ArgumentList $smiArgs `
    -RedirectStandardOutput $timelineCsv `
    -NoNewWindow `
    -PassThru

# Sleep for the requested duration. WarmupSkipSeconds is informational
# (aggregator drops first N seconds); the monitor itself runs full window.
$startUtc = (Get-Date).ToUniversalTime()
Start-Sleep -Seconds $DurationSeconds
$endUtc = (Get-Date).ToUniversalTime()

# Stop nvidia-smi gracefully.
if (-not $smiProc.HasExited) {
    Stop-Process -Id $smiProc.Id -Force
    $smiProc.WaitForExit(5000) | Out-Null
}

Write-Host "[utilization_monitor] nvidia-smi stopped, timeline=$timelineCsv"

# Optional: ncu single-point sampling.
$ncuRan = $false
if ($NcuTargetExe) {
    $ncu = Get-Command ncu -ErrorAction SilentlyContinue
    if (-not $ncu) {
        Write-Warning "[utilization_monitor] NcuTargetExe given but ncu not in PATH; skipping per-kernel sampling"
    }
    elseif (-not (Test-Path -LiteralPath $NcuTargetExe)) {
        Write-Warning "[utilization_monitor] NcuTargetExe not found: $NcuTargetExe; skipping per-kernel sampling"
    }
    else {
        $metrics = @(
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__t_sectors_aperture_device_op_read.sum",
            "l1tex__t_bytes.sum"
        ) -join ","

        $ncuArgs = @(
            "--metrics", $metrics,
            "--target-processes", "all",
            "--csv",
            "--page", "raw",
            "--launch-skip", "5",
            "--launch-count", "10",
            "-o", "/dev/null",  # discard report file, we only want stdout CSV
            $NcuTargetExe
        )
        if ($NcuTargetArgs) {
            $extra = $NcuTargetArgs.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
            $ncuArgs = $ncuArgs + $extra
        }

        Write-Host "[utilization_monitor] launching ncu single-point sampling -> $ncuOut"
        & ncu @ncuArgs > $ncuOut 2>&1
        $ncuRan = $true
    }
}

# Write meta.json with run parameters + timestamps.
$meta = [pscustomobject]@{
    run_id              = $RunId
    duration_seconds    = $DurationSeconds
    interval_ms         = $IntervalMs
    warmup_skip_seconds = $WarmupSkipSeconds
    start_utc           = $startUtc.ToString("o")
    end_utc             = $endUtc.ToString("o")
    timeline_csv        = $timelineCsv
    ncu_metrics_txt     = if ($ncuRan) { $ncuOut } else { $null }
    ncu_target_exe      = $NcuTargetExe
    ncu_target_args     = $NcuTargetArgs
    query_fields        = $queryFields
}
$meta | ConvertTo-Json -Depth 4 | Set-Content -Path $metaJson -Encoding utf8

Write-Host "[utilization_monitor] done. meta=$metaJson"
Write-Output $runDir
