#!/usr/bin/env bash
# Run from any directory. Keep setup chatter off stdout for --json and --markdown.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

UV_BIN="$(command -v uv || true)"
for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" "$SCRIPT_DIR/.local/bin/uv"; do
    if [ -z "$UV_BIN" ] && [ -x "$candidate" ]; then
        UV_BIN="$candidate"
    fi
done

# Prefer an existing Python. uv can supply one if the host has none.
PYTHON_BIN=""
for candidate in "$SCRIPT_DIR/.venv/bin/python" python3 python; do
    if "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    if [ -z "$UV_BIN" ]; then
        if ! command -v curl >/dev/null 2>&1; then
            echo "ModelScout needs Python 3.11+, uv, or curl to set up Python." >&2
            exit 1
        fi
        echo "Installing uv locally so it can set up Python..." >&2
        mkdir -p "$SCRIPT_DIR/.local/bin"
        curl --fail --silent --show-error --location https://astral.sh/uv/install.sh \
            --output "$SCRIPT_DIR/.local/uv-installer.sh"
        UV_INSTALL_DIR="$SCRIPT_DIR/.local/bin" UV_NO_MODIFY_PATH=1 \
            sh "$SCRIPT_DIR/.local/uv-installer.sh" >&2
        UV_BIN="$SCRIPT_DIR/.local/bin/uv"
    fi
    "$UV_BIN" python install 3.12 >&2
    PYTHON_BIN="$("$UV_BIN" python find 3.12)"
fi

export MODELSCOUT_UV="$UV_BIN"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [ "$#" -eq 0 ]; then
    echo "Starting ModelScout hardware scan..." >&2
    echo "For the web dashboard, run: ./start.sh web" >&2
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/launch.py" "$@"
