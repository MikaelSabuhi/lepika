#!/bin/sh
# LePika installer — installs uv (if needed) and lepika, then starts setup.
set -eu

PACKAGE="lepika"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# uv installs both itself and its tools into ~/.local/bin. Added unconditionally,
# so the lepika just installed below is runnable even when uv was already present.
PATH="$HOME/.local/bin:$PATH"
export PATH

echo "Installing LePika…"
uv tool install --force "$PACKAGE"

echo ""
echo "✓ LePika installed."
echo 'Note: lepika lives in ~/.local/bin. If a NEW terminal says "lepika: command'
echo 'not found", add that directory to your PATH, or just restart your shell.'
echo ""

if ! command -v lepika >/dev/null 2>&1; then
  echo "lepika is installed but not on this shell's PATH."
  echo "Open a new terminal and run: lepika"
  exit 0
fi

echo "Starting setup…"
# < /dev/tty: piped from curl, this script's stdin is the (now exhausted) pipe,
# so the wizard's prompts would read EOF. Read from the terminal instead.
exec lepika < /dev/tty
