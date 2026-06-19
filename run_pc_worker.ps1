$ErrorActionPreference = "Stop"

$env:JARVIS_SERVER_URL = if ($env:JARVIS_SERVER_URL) { $env:JARVIS_SERVER_URL } else { "https://jarvis-railway-production-42ff.up.railway.app" }
$env:JARVIS_WORK_DIR = if ($env:JARVIS_WORK_DIR) { $env:JARVIS_WORK_DIR } else { "$HOME\Documents\Jarvis Work" }
$env:JARVIS_WORKER_INTERVAL = if ($env:JARVIS_WORKER_INTERVAL) { $env:JARVIS_WORKER_INTERVAL } else { "20" }

Write-Host "Starting PC JARVIS worker loop..."
Write-Host "Server: $env:JARVIS_SERVER_URL"
Write-Host "Work folder: $env:JARVIS_WORK_DIR"
Write-Host "Interval: $env:JARVIS_WORKER_INTERVAL seconds"
Write-Host ""

python pc_jarvis.py --poll
