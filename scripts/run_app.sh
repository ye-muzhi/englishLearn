#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

case "$(uname -s)" in
  Darwin) DATA_HOME="$HOME/Library/Application Support/EnglishLearn" ;;
  *) DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/EnglishLearn" ;;
esac
if [[ -z "${ENGLISHLEARN_WORK_DIR:-}" ]] && \
   [[ -f "$ROOT/work/projects.json" || -d "$ROOT/work/models" ]]; then
  DATA_HOME="$ROOT/work"
fi

PORT="${ENGLISHLEARN_PORT:-8501}"
URL="http://localhost:${PORT}"
PYTHON="$ROOT/.venv/bin/python"

open_app() {
  if [[ "${ENGLISHLEARN_NO_OPEN:-0}" == "1" ]]; then
    echo "EnglishLearn is ready: $URL"
  elif command -v open >/dev/null 2>&1; then
    open "$URL"
  else
    echo "EnglishLearn is ready: $URL"
  fi
}

if [[ ! -x "$PYTHON" ]]; then
  echo "EnglishLearn runtime was not found."
  echo "Run: $ROOT/scripts/setup_local.sh"
  read -r -p "Press Enter to close..." _ </dev/tty || true
  exit 1
fi

if curl -fsS "$URL/_stcore/health" >/dev/null 2>&1; then
  echo "EnglishLearn is already running."
  open_app
  exit 0
fi

export ENGLISHLEARN_WORK_DIR="${ENGLISHLEARN_WORK_DIR:-$DATA_HOME}"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -B -m streamlit run "$ROOT/app.py" \
  --server.port "$PORT" \
  --server.fileWatcherType none &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup INT TERM EXIT

for _ in {1..40}; do
  if curl -fsS "$URL/_stcore/health" >/dev/null 2>&1; then
    echo "EnglishLearn started successfully."
    open_app
    wait "$SERVER_PID"
    STATUS=$?
    trap - INT TERM EXIT
    exit "$STATUS"
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    wait "$SERVER_PID"
    exit $?
  fi
  sleep 0.2
done

echo "EnglishLearn did not become ready within 8 seconds."
cleanup
wait "$SERVER_PID" 2>/dev/null || true
exit 1
