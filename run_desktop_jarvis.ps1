$ErrorActionPreference = "Stop"

$env:JARVIS_SERVER_URL = if ($env:JARVIS_SERVER_URL) { $env:JARVIS_SERVER_URL } else { "https://jarvis-railway-production-42ff.up.railway.app" }
$env:JARVIS_WORK_DIR = if ($env:JARVIS_WORK_DIR) { $env:JARVIS_WORK_DIR } else { "$HOME\Documents\Jarvis Work" }

Set-Location -LiteralPath $PSScriptRoot
Write-Host "Starting Desktop JARVIS..."
Write-Host "Server: $env:JARVIS_SERVER_URL"
Write-Host "Work folder: $env:JARVIS_WORK_DIR"
python desktop_jarvis.py
