# ezai installer — installs uv (if needed) and ezai, then starts setup.
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python package manager)…"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Installing ezai…"
uv tool install --force ezai
# $ErrorActionPreference does not trip on a native command's exit code, so a
# failed install would otherwise fall through to the success message below.
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "✓ ezai installed. Starting setup…"
uv tool run ezai
