#!/bin/sh
# ezai installer — installs uv (if needed) and ezai, then starts setup.
set -eu

REPO_URL="git+https://github.com/MikaelSabuhi/ezaiselfhost"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# uv installs both itself and its tools into ~/.local/bin. Added unconditionally,
# so the ezai just installed below is runnable even when uv was already present.
PATH="$HOME/.local/bin:$PATH"
export PATH

echo "Installing ezai…"
# Installed from this repository by name, never from a bare package name: the
# name `ezai` on PyPI belongs to an unrelated third party, and resolving it here
# would run someone else's code on a machine that asked for ours.
uv tool install --force "$REPO_URL"

echo ""
echo "✓ ezai installed."
echo 'Note: ezai lives in ~/.local/bin. If a NEW terminal says "ezai: command'
echo 'not found", add that directory to your PATH, or just restart your shell.'
echo ""

# Run the installed executable directly rather than via `uv tool run`, which
# would resolve the bare name `ezai` against PyPI if it were ever missing here.
if ! command -v ezai >/dev/null 2>&1; then
  echo "ezai is installed but not on this shell's PATH."
  echo "Open a new terminal and run: ezai"
  exit 0
fi

echo "Starting setup…"
# < /dev/tty: piped from curl, this script's stdin is the (now exhausted) pipe,
# so the wizard's prompts would read EOF. Read from the terminal instead.
exec ezai < /dev/tty
