#!/usr/bin/env bash
#
# Launches the MeetSense Streamlit app using the project's .venv
# ONLY. Never falls back to conda base or any other interpreter on
# PATH -- it invokes .venv's python by absolute path throughout, so
# PATH ordering (conda, homebrew, system python, etc.) cannot affect
# which Streamlit actually runs.
#
# Usage: ./run_app.sh [extra streamlit args]
#   e.g. ./run_app.sh --server.headless true

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "ERROR: $VENV_PYTHON not found or not executable." >&2
    echo "Expected a working .venv at $SCRIPT_DIR/.venv -- see PROJECT_REPORT for setup." >&2
    exit 1
fi

# Sanity check: confirm this interpreter actually resolves into the
# project's own .venv/lib site-packages, not conda base or anywhere
# else, before we launch anything.
VENV_CHECK=$("$VENV_PYTHON" -c "import sys; print(sys.prefix)")
case "$VENV_CHECK" in
    "$SCRIPT_DIR"/.venv*) ;;
    *)
        echo "ERROR: $VENV_PYTHON resolved to sys.prefix=$VENV_CHECK," >&2
        echo "not $SCRIPT_DIR/.venv -- refusing to launch with the wrong environment." >&2
        exit 1
        ;;
esac

# MPS (Apple GPU) backend doesn't implement every torch op used by
# Whisper/pyannote/T5; this lets unsupported ops silently fall back
# to CPU instead of crashing, rather than requiring every op to be
# verified MPS-complete first.
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "============================================================"
echo "MeetSense -- launching with:"
echo "  $VENV_PYTHON"
echo "  (sys.prefix: $VENV_CHECK)"
echo "============================================================"
echo ""
echo "This machine has 8GB RAM. Whisper + pyannote + T5 loaded"
echo "together are memory-heavy -- CLOSE other heavy apps first"
echo "(Chrome, other IDEs/extra VS Code windows, etc.) or the Live"
echo "Meeting tab's first model load may hang for several minutes"
echo "or longer under memory pressure. This is a known, measured"
echo "constraint of this hardware, not a bug to retry past."
echo "============================================================"
echo ""

exec "$VENV_PYTHON" -m streamlit run "$SCRIPT_DIR/app_workspace.py" "$@"
