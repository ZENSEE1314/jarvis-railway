$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

Write-Host "Building Desktop-JARVIS.exe..."

$pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (Get-Command py -ErrorAction SilentlyContinue) {
  $pyList = & py -0p
  $python311 = $pyList | Where-Object { $_ -match "3\.11" } | Select-Object -First 1
  if ($python311 -and $python311 -match "([A-Z]:\\.*python\.exe)") {
    $pythonExe = $Matches[1]
  }
}

if (-not $pythonExe) {
  throw "Python was not found. Install Python 3.11+ first."
}

Write-Host "Using Python: $pythonExe"

function Invoke-Python {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  & $pythonExe @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed: $pythonExe $($Args -join ' ')"
  }
}

$buildVenv = Join-Path $PSScriptRoot ".venv-build"
if (-not (Test-Path $buildVenv)) {
  Invoke-Python -m venv $buildVenv
}

$venvPython = Join-Path $buildVenv "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  throw "Build virtual environment was not created."
}

function Invoke-BuildPython {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  & $venvPython @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Build Python command failed: $venvPython $($Args -join ' ')"
  }
}

Invoke-BuildPython -m pip install --upgrade pip
Invoke-BuildPython -m pip install pyinstaller

Invoke-BuildPython -m PyInstaller `
  --noconfirm `
  --clean `
  --name "Desktop-JARVIS" `
  --console `
  --add-data "desktop.html;." `
  desktop_jarvis.py

if (-not (Test-Path "$PSScriptRoot\dist\Desktop-JARVIS\Desktop-JARVIS.exe")) {
  throw "Build finished but Desktop-JARVIS.exe was not found."
}

Write-Host ""
Write-Host "Done. EXE created at:"
Write-Host "$PSScriptRoot\dist\Desktop-JARVIS\Desktop-JARVIS.exe"
