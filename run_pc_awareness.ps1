$ErrorActionPreference = "Stop"

$env:JARVIS_SERVER_URL = if ($env:JARVIS_SERVER_URL) { $env:JARVIS_SERVER_URL } else { "https://jarvis-railway-production-42ff.up.railway.app" }
$env:JARVIS_WORKER_INTERVAL = if ($env:JARVIS_WORKER_INTERVAL) { $env:JARVIS_WORKER_INTERVAL } else { "15" }

Write-Host "Starting PC JARVIS awareness mode..."
Write-Host "This logs foreground window titles only. It does not record keystrokes, passwords, or screenshots."
Write-Host ""

python pc_jarvis.py --awareness
