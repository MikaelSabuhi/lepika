# ezai installer — installs uv (if needed) and ezai, then starts setup.
$ErrorActionPreference = "Stop"

$RepoUrl = "git+https://github.com/MikaelSabuhi/ezaiselfhost"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python package manager)…"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}

# uv installs both itself and its tools here. Added unconditionally, so the ezai
# installed below is runnable even when uv was already present.
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

Write-Host "Installing ezai…"
# Installed from this repository by name, never from a bare package name: the
# name `ezai` on PyPI belongs to an unrelated third party, and resolving it here
# would run someone else's code on a machine that asked for ours.
uv tool install --force $RepoUrl
# $ErrorActionPreference does not trip on a native command's exit code, so a
# failed install would otherwise fall through to the success message below.
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "✓ ezai installed."
Write-Host 'Note: ezai lives in ~\.local\bin. If a NEW terminal cannot find'
Write-Host '"ezai", add that directory to your PATH, or restart your shell.'
Write-Host ""

# Run the installed executable directly rather than via `uv tool run`, which
# would resolve the bare name `ezai` against PyPI if it were ever missing here.
if (-not (Get-Command ezai -ErrorAction SilentlyContinue)) {
    Write-Host "ezai is installed but not on this shell's PATH."
    Write-Host "Open a new terminal and run: ezai"
    exit 0
}

Write-Host "Starting setup…"
ezai
