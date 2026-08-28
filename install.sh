#!/bin/sh
# ezai installer — installs uv (if needed) and ezai, then starts setup.
set -eu

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs into ~/.local/bin
  PATH="$HOME/.local/bin:$PATH"
  export PATH
fi

echo "Installing ezai…"
uv tool install --force ezai

echo ""
echo "✓ ezai installed. Starting setup…"
# < /dev/tty: piped from curl, this script's stdin is the (now exhausted) pipe,
# so the wizard's prompts would read EOF. Read from the terminal instead.
exec uv tool run ezai < /dev/tty
