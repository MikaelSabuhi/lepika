# LePika installer — installs uv (if needed) and lepika, then starts setup.
$ErrorActionPreference = "Stop"

$RepoUrl = "git+https://github.com/MikaelSabuhi/lepika"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python package manager)…"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}

# uv installs both itself and its tools here. Added unconditionally, so the lepika
# installed below is runnable even when uv was already present.
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

Write-Host "Installing LePika…"
# Installed from this repository by URL, never from a bare package name: LePika
# is not published to PyPI, and resolving a bare name there would run someone
# else's code on a machine that asked for ours.
uv tool install --force $RepoUrl
# $ErrorActionPreference does not trip on a native command's exit code, so a
# failed install would otherwise fall through to the success message below.
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "✓ LePika installed."
Write-Host 'Note: lepika lives in ~\.local\bin. If a NEW terminal cannot find'
Write-Host '"lepika", add that directory to your PATH, or restart your shell.'
Write-Host ""

# Run the installed executable directly rather than via `uv tool run`, which
# would resolve the bare name `lepika` against PyPI if it were ever missing here.
if (-not (Get-Command lepika -ErrorAction SilentlyContinue)) {
    Write-Host "lepika is installed but not on this shell's PATH."
    Write-Host "Open a new terminal and run: lepika"
    exit 0
}

Write-Host "Starting setup…"
lepika
