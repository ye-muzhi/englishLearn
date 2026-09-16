#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$ROOT/scripts/setup_local.sh"
fi
exec "$ROOT/scripts/run_app.sh"
