#!/usr/bin/env bash
# Start the listen-demo service on 127.0.0.1 (override with HOST/PORT).
set -euo pipefail
cd "$(dirname "$0")"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8766}"

pick_python() {
  if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
    echo "${PYTHON}"
    return
  fi
  if [[ -x .venv/bin/python ]]; then
    echo .venv/bin/python
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "未找到 python3。请先安装 Python 3。" >&2
  exit 1
}

PY="$(pick_python)"
echo "using ${PY}"
# If cache is already built, stdlib-only Python can still serve the UI.
# First-time precompute and POST /api/cancel need numpy + scipy.
exec "${PY}" server.py --host "${HOST}" --port "${PORT}" "$@"
