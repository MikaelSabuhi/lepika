#!/usr/bin/env bash
# LePika smoke test: the real thing on a real machine.
#
#   .github/scripts/smoke.sh express|server
#
# Runs the wizard non-interactively, asks the engine one question, exercises
# expose (Server), down and up, then leaves the machine as it found it — bar the
# Ollama that `lepika down` keeps running on purpose. Needs `lepika` on PATH,
# curl, and a scratch LEPIKA_HOME (one is created when unset).
set -euo pipefail

mode="${1:?usage: smoke.sh express|server}"
model="qwen3:0.6b"
engine="http://127.0.0.1:11434"
webui="http://127.0.0.1:3000"
api="http://127.0.0.1:11435"

export LEPIKA_HOME="${LEPIKA_HOME:-$(mktemp -d)}"
# The wizard ends by opening the chat UI; `true` is a browser that swallows the URL.
export BROWSER=true

say() { printf '\n== %s\n' "$*"; }

# wait_status URL CODE [SECONDS]: poll until URL answers with CODE (000 = refused).
wait_status() {
  local url="$1" want="$2" seconds="${3:-30}" got=""
  for _ in $(seq "$seconds"); do
    got=$(curl -s -o /dev/null -w '%{http_code}' "$url" || true)
    [ "$got" = "$want" ] && return 0
    sleep 1
  done
  echo "expected HTTP $want from $url, got $got" >&2
  return 1
}

say "wizard: $mode mode, $model, LEPIKA_HOME=$LEPIKA_HOME"
printf '%s\n' "$model" | lepika --mode "$mode"

say "status, model list, doctor"
lepika status
lepika model list | grep -F "$model"
lepika doctor

say "chat completion"
reply=$(curl -sfS "$engine/api/chat" -d "{\"model\":\"$model\",\"think\":false,\"stream\":false,
  \"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word: pika\"}]}")
echo "$reply"
echo "$reply" | grep -q '"done":true'
wait_status "$webui/health" 200 5

if [ "$mode" = server ]; then
  say "expose: key required on $api"
  # The key is printed by `lepika expose`; keep it out of the job log and mask it
  # before anything else can echo it.
  lepika expose >/dev/null
  key=$(sed -n "s/^LEPIKA_API_KEY='\(.*\)'$/\1/p" "$LEPIKA_HOME/stack/.env")
  [ -n "$key" ]
  echo "::add-mask::$key"
  wait_status "$api/api/version" 401
  curl -sfS -H "Authorization: Bearer $key" "$api/api/version"
  echo
  lepika expose --off
  wait_status "$api/api/version" 000
fi

say "down, then up again"
lepika down
wait_status "$webui/health" 000
lepika up
wait_status "$webui/health" 200 5
lepika down

say "ok"
