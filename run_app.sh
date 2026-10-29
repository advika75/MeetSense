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

# src/diarization.py needs Homebrew ffmpeg on DYLD_LIBRARY_PATH for
# pyannote's torchcodec-based audio loader, and sets it itself via
# os.execve() (re-exec is the only way to make a dlopen()-time env var
# change take effect) if _MEETSENSE_DYLD_FIXED isn't already "1". That
# self-correcting re-exec is harmless for one-shot batch scripts, but
# fatal here: diarization.py is imported lazily, the first time a Live
# Meeting session is started, and os.execve() replaces this *entire
# running Streamlit server process* in place -- killing every open
# browser WebSocket mid-session (the "Connection error" users see).
# Setting both env vars before Streamlit ever starts satisfies that
# guard up front, so the re-exec never fires once the server is live.
FFMPEG_LIB="/opt/homebrew/opt/ffmpeg/lib"
if [ -d "$FFMPEG_LIB" ]; then
    export DYLD_LIBRARY_PATH="$FFMPEG_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export _MEETSENSE_DYLD_FIXED=1
fi

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
