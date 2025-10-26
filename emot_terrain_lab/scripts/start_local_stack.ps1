param(
    [switch]$NoWatcher,
    [switch]$NoHotkeys,
    [switch]$NoDashboard,
    [string]$LogLevel = "INFO"
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
$ActivateScript = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
$HasVenv = Test-Path $ActivateScript

function Start-Component {
    param(
        [string]$Title,
        [string]$Command
    )

    $script = @"
& {
    `$Host.UI.RawUI.WindowTitle = '$Title'
    Set-Location '$ProjectRoot'
    if (Test-Path '$ActivateScript') {
        . '$ActivateScript'
    }
    `$env:PYTHONPATH = '$ProjectRoot'
    $Command
}
"@
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $script -WorkingDirectory $ProjectRoot | Out-Null
}

Write-Host "== EQNet local stack ==" -ForegroundColor Cyan
if ($HasVenv) {
    Write-Host "Using virtual environment (.venv) for each window." -ForegroundColor DarkGray
} else {
    Write-Host "Virtual environment (.venv) not found; using system Python." -ForegroundColor DarkYellow
}

Start-Component "EQNet Bus" "python ops\hub_ws.py --log-level $LogLevel"

if (-not $NoWatcher) {
    Start-Component "EQNet Config Watcher" "python ops\config_watcher.py"
}

if (-not $NoHotkeys) {
    Start-Component "EQNet Hotkeys" "python ops\hotkeys.py"
}

if (-not $NoDashboard) {
    Start-Component "EQNet Dashboard" "python ops\dashboard.py"
}

Write-Host "Started. Close the spawned PowerShell windows or press Ctrl+C in each to stop them." -ForegroundColor Cyan

