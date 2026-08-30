# LePika installer — installs uv (if needed) and lepika, then starts setup.
$ErrorActionPreference = "Stop"

$Package = "lepika"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python package manager)…"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}

# uv installs both itself and its tools here. Added unconditionally, so the lepika
# installed below is runnable even when uv was already present.
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

Write-Host "Installing LePika…"
uv tool install --force $Package
# $ErrorActionPreference does not trip on a native command's exit code, so a
# failed install would otherwise fall through to the success message below.
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "✓ LePika installed."
Write-Host 'Note: lepika lives in ~\.local\bin. If a NEW terminal cannot find'
Write-Host '"lepika", add that directory to your PATH, or restart your shell.'
Write-Host ""

if (-not (Get-Command lepika -ErrorAction SilentlyContinue)) {
    Write-Host "lepika is installed but not on this shell's PATH."
    Write-Host "Open a new terminal and run: lepika"
    exit 0
}

Write-Host "Starting setup…"
lepika
