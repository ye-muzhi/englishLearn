#!/usr/bin/env bash
set -euo pipefail

# Creates an isolated Python 3.11 runtime for the app. It never modifies the
# system Python installation. Add --download-model to fetch Hy-MT2-1.8B.
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
export ENGLISHLEARN_WORK_DIR="${ENGLISHLEARN_WORK_DIR:-$DATA_HOME}"

TOOLS_DIR="$ROOT/.tools"
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$TOOLS_DIR/uv"
fi
if [[ ! -x "$UV_BIN" ]]; then
  mkdir -p "$TOOLS_DIR"
  echo "Preparing the EnglishLearn installer..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$TOOLS_DIR" sh
fi

"$UV_BIN" venv --allow-existing --python 3.11 .venv
"$UV_BIN" pip install --python .venv/bin/python -r requirements.txt

if [[ "${1:-}" == "--download-model" ]]; then
  .venv/bin/python -m englishlearn.translation.hy_mt2_local --download
fi

echo "Setup complete. Double-click EnglishLearn.command or run scripts/run_app.sh"
