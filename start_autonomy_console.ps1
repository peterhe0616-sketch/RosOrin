param(
    [string]$RobotHost = "192.168.3.17",
    [switch]$NoVlm
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv-vla\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Run setup_vla.ps1 first. Python environment not found: $Python"
}

$Arguments = @("autonomy_console.py", "--host", $RobotHost)
if ($NoVlm) { $Arguments += "--no-vlm" }
& $Python @Arguments

