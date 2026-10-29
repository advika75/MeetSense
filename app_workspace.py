import contextlib
import json
import os
import subprocess
import textwrap
import sys
import time
import threading
from pathlib import Path

import streamlit as st

_SRC_DIR = Path(__file__).resolve().parent / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Local sounddevice mic path (Stage 4), not the browser/webrtc path --
# live_web.py and streamlit_webrtc remain unused here but untouched on
# disk as prior art. live_controller.py owns a worker thread that
# calls into live_pipeline.py's LiveMeetingSession; this script must
# never call VAD/ASR/diarization itself, only read controller state.
#
# NOT imported here at module level, deliberately: live_controller
# imports live_pipeline, which imports vad/transcription/diarization/
# summarization, which import torch/whisper/pyannote/transformers at
# THEIR module level. If this file imported live_controller here, every
# page load -- Dashboard, batch results, anything -- would pay that
# import cost (minutes, under memory pressure on this 8GB machine)
# before rendering a single pixel. Imported lazily instead, inside the
# Live Meeting page body, only once the user clicks Start.


def _refine_live_report_speakers(report_id, audio_path, state):
    """Run the opt-in offline pyannote pass without blocking Streamlit."""
    try:
        from diarization import load_diarization_pipeline, diarize_audio
        from merge_transcript import align_segments
        from meeting_report import load_report, save_report, compute_speaking_time
        import gc

        pipeline = load_diarization_pipeline()
        diarized = diarize_audio(pipeline, audio_path)
        report = load_report(report_id)
        report.transcript = align_segments(report.transcript, diarized)
        report.speakers = compute_speaking_time(report.transcript)
        report.duration_sec = max((row["end"] for row in report.transcript), default=0.0)
        save_report(report)
        pipeline = None
        gc.collect()
        state.update({"running": False, "done": True, "error": None})
    except Exception as error:
        state.update({"running": False, "done": False, "error": str(error)})


# ============================================================
# THEME
# ============================================================
#
# Single source of truth for the whole app: st.session_state.dark_mode
# (a plain boolean, driven by exactly one st.toggle -- see "THEME
# SWITCH" below). Every other former theme mechanism (a separate
# unused "theme" string, an OS-prefers-color-scheme-only CSS block, a
# second sidebar radio control that had no visual effect, and a
# duplicated set of Python color constants) has been removed so
# nothing can fall out of sync with this one flag.

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

THEMES = {
    "light": {
        "bg": "#F7F7F4",
        "bg_secondary": "#EFEEEA",
        "sidebar": "#FFFFFF",
        "card": "#FFFFFF",
        "text": "#1C1E22",
        "text_secondary": "#454850",
        "muted": "#6B6E76",
        "border": "#DCDDD9",
        "accent": "#1F6F5C",
        "accent_hover": "#175A4A",
        "accent_soft": "#E3EEEA",
        "accent_solid": "#1F6F5C",
        "accent_solid_hover": "#175A4A",
        "success": "#1E7D46",
        "success_bg": "#E7F4EA",
        "warning": "#8A5A00",
        "warning_bg": "#FDF1DD",
        "danger": "#B23B2E",
        "danger_bg": "#FBE9E6",
        "input": "#FFFFFF",
        "sidebar_hover": "#F0F3F1",
        "code_bg": "#F1F1EE",
        "shadow": "0 1px 2px rgba(20,20,20,0.06)",
    },
    "dark": {
        "bg": "#15171A",
        "bg_secondary": "#1B1D21",
        "sidebar": "#121315",
        "card": "#1B1D21",
        "text": "#EAEBEC",
        "text_secondary": "#C2C4C8",
        "muted": "#8C8F95",
        "border": "#2B2D31",
        "accent": "#4FA68C",
        "accent_hover": "#63BBA0",
        "accent_soft": "#1C2B27",
        "accent_solid": "#2F8A71",
        "accent_solid_hover": "#3A9C80",
        "success": "#4FBE85",
        "success_bg": "#15271C",
        "warning": "#E0B256",
        "warning_bg": "#2E2410",
        "danger": "#E2776A",
        "danger_bg": "#311A16",
        "input": "#1D1F23",
        "sidebar_hover": "#1D2521",
        "code_bg": "#101214",
        "shadow": "0 2px 10px rgba(0,0,0,0.3)",
    },
}

theme = THEMES["dark" if st.session_state.dark_mode else "light"]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MeetSense — Meeting Intelligence",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME VARIABLES (single CSS injection point)
# ============================================================
# Defines --bg-primary etc. as CSS custom properties from the THEMES
# dict above. Every component rule below (and throughout the file)
# reads these variables instead of hardcoding colors, so a single
# dark_mode flip updates everything at once.

st.markdown(
    f"""
<style>
:root {{
    --bg-primary: {theme['bg']};
    --bg-secondary: {theme['bg_secondary']};
    --bg-card: {theme['card']};
    --bg-sidebar: {theme['sidebar']};
    --sidebar-hover: {theme['sidebar_hover']};
    --text-primary: {theme['text']};
    --text-secondary: {theme['text_secondary']};
    --text-muted: {theme['muted']};
    --border: {theme['border']};
    --accent: {theme['accent']};
    --accent-hover: {theme['accent_hover']};
    --accent-solid: {theme['accent_solid']};
    --accent-solid-hover: {theme['accent_solid_hover']};
    --accent-soft: {theme['accent_soft']};
    --success: {theme['success']};
    --success-bg: {theme['success_bg']};
    --warning: {theme['warning']};
    --warning-bg: {theme['warning_bg']};
    --error: {theme['danger']};
    --error-bg: {theme['danger_bg']};
    --input-bg: {theme['input']};
    --input-text: {theme['text']};
    --code-bg: {theme['code_bg']};
    --shadow: {theme['shadow']};
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "data" / "audio"
OUTPUT_DIR = BASE_DIR / "outputs"

TRANSCRIPT_DIR = OUTPUT_DIR / "transcripts"
NLP_DIR = OUTPUT_DIR / "nlp"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
EVAL_DIR = OUTPUT_DIR / "evaluations"


AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

SPEAKER_TRANSCRIPT = TRANSCRIPT_DIR / "speaker_transcript.json"
MEETING_TRANSCRIPT = TRANSCRIPT_DIR / "meeting.json"
DIARIZATION = TRANSCRIPT_DIR / "diarization.json"

# Experimental diarization backend (see README "Experimental Diarization
# Comparison" section) -- separate output files, never overwriting the
# production PyAnnote files above.
DIARIZATION_VBX = TRANSCRIPT_DIR / "diarization_vbx.json"

NLP_PROCESSED = NLP_DIR / "nlp_processed_transcript.json"
TFIDF = NLP_DIR / "tfidf_keywords.json"
TOPICS = NLP_DIR / "topic_modeling.json"
ACTION_ITEMS = NLP_DIR / "action_items_decisions.json"

SUMMARY = SUMMARY_DIR / "meeting_summary.json"

WER = EVAL_DIR / "wer_results.json"
DER = EVAL_DIR / "der_results.json"
DER_VBX = EVAL_DIR / "der_vbx.json"
ROUGE = EVAL_DIR / "rouge_results.json"
ACTION_EVAL = EVAL_DIR / "action_item_results.json"

MODEL_COMPARISON = EVAL_DIR / "model_comparison.json"
ABLATION_RESULTS = EVAL_DIR / "ablation_results.json"
ERROR_ANALYSIS = EVAL_DIR / "error_analysis.json"
SPEAKER_ANALYSIS = EVAL_DIR / "speaker_analysis.json"
EVIDENCE_LINKED_ITEMS = EVAL_DIR / "evidence_linked_items.json"


# ============================================================
# CUSTOM CSS
# ============================================================
#
# One static stylesheet. Every rule below reads the --bg-primary /
# --text-primary / etc. custom properties defined above (from
# THEMES + st.session_state.dark_mode) instead of hardcoding colors,
# so flipping dark_mode updates every component at once with no
# separate light/dark rule sets to keep in sync.

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg-primary);
    color: var(--text-primary);
}

/* Hide Streamlit default UI */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header {
    visibility: visible !important;
    background: transparent !important;
}

/* Main container */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* ------------------------------------------------------------
   Sidebar
   ------------------------------------------------------------ */

section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label * {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}

/* Nav rows get a hover affordance and a clearly-visible selected state */
section[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 8px;
    padding: 2px 8px;
    transition: background-color .12s ease;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: var(--sidebar-hover) !important;
}

section[data-testid="stSidebar"] [role="radio"][aria-checked="true"],
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] * {
    color: var(--accent) !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
    background: var(--accent-soft) !important;
    border-radius: 8px;
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: var(--text-muted) !important;
    opacity: 1 !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapsedControl"] button {
    color: var(--text-primary) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: var(--shadow) !important;
}

[data-testid="stSidebarCollapsedControl"] svg {
    color: var(--text-primary) !important;
    fill: var(--text-primary) !important;
}

/* ------------------------------------------------------------
   Brand / top bar / titles
   ------------------------------------------------------------ */

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 5px 8px 25px 8px;
}

.brand-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: var(--accent-solid);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    flex-shrink: 0;
}

.brand-name {
    font-size: 20px;
    font-weight: 800;
    color: var(--text-primary) !important;
}

.topbar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 13px 18px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    box-shadow: var(--shadow);
}

.top-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
}

.page-title {
    font-size: 31px;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 4px;
}

.page-subtitle {
    color: var(--text-secondary);
    font-size: 14px;
    margin-bottom: 25px;
}

.section-heading {
    font-size: 18px;
    font-weight: 750;
    color: var(--text-primary);
    margin-top: 12px;
    margin-bottom: 12px;
}

.soft-divider {
    height: 1px;
    background: var(--border);
    margin: 18px 0;
}

/* ------------------------------------------------------------
   Cards
   ------------------------------------------------------------ */

.card,
.meeting-header {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: var(--shadow);
}

.card-title,
.meeting-name {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 12px;
}

.card-subtitle,
.meeting-meta {
    font-size: 12px;
    color: var(--text-muted);
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 19px;
    min-height: 120px;
    box-shadow: var(--shadow);
}

.stat-icon { font-size: 20px; margin-bottom: 10px; }

.stat-number {
    font-size: 27px;
    font-weight: 800;
    color: var(--text-primary);
}

.stat-label {
    color: var(--text-muted);
    font-size: 12px;
    margin-top: 3px;
}

/* ------------------------------------------------------------
   Speakers / transcript
   ------------------------------------------------------------ */

.speaker-row {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 13px 15px;
    margin-bottom: 10px;
}

.speaker-name {
    font-size: 14px;
    font-weight: 700;
    color: var(--text-primary);
}

.speaker-time {
    font-size: 12px;
    color: var(--text-muted);
}

.transcript-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 15px 17px;
    margin-bottom: 10px;
    transition: border-color .15s ease, box-shadow .15s ease;
}

.transcript-item:hover {
    border-color: var(--accent);
    box-shadow: var(--shadow);
}

.transcript-header {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
}

.transcript-speaker {
    font-size: 13px;
    font-weight: 700;
    color: var(--accent);
}

.transcript-time {
    font-size: 11px;
    color: var(--text-muted);
}

.transcript-text {
    font-size: 14px;
    line-height: 1.65;
    color: var(--text-primary);
    word-wrap: break-word;
    overflow-wrap: break-word;
}

/* ------------------------------------------------------------
   Tags / status / evaluation metrics
   ------------------------------------------------------------ */

.tag {
    display: inline-block;
    padding: 7px 11px;
    margin: 4px 4px 4px 0;
    background: var(--accent-soft);
    color: var(--accent);
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

.status-pill {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 20px;
    background: var(--success-bg);
    color: var(--success);
    font-size: 11px;
    font-weight: 700;
}

.metric {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 15px;
    text-align: center;
}

.metric-name {
    color: var(--text-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 25px;
    font-weight: 800;
    color: var(--accent);
    margin-top: 5px;
}

/* ------------------------------------------------------------
   Action items / decisions / summary
   ------------------------------------------------------------ */

.action-item {
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 9px;
    background: var(--bg-card);
}

.action-title {
    font-weight: 600;
    font-size: 13px;
    color: var(--text-primary);
}

.decision {
    background: var(--accent-soft);
    border-left: 4px solid var(--accent);
    padding: 14px 16px;
    border-radius: 9px;
    margin-bottom: 10px;
    font-size: 13px;
    line-height: 1.5;
    color: var(--text-primary);
}

.summary-box {
    background: var(--accent-soft);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-primary);
}

/* ------------------------------------------------------------
   Native Streamlit widgets
   ------------------------------------------------------------ */

h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }
p, li, label { color: var(--text-secondary); }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: var(--text-primary);
}

.stButton > button,
.stDownloadButton > button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    /* accent-solid (not accent) -- verified >=4.5:1 with white text in
       both themes. --accent itself is deliberately lighter in dark
       mode (for readable text on dark backgrounds), which would fail
       contrast as a button fill with white text on top. */
    background: var(--accent-solid) !important;
    color: #ffffff !important;
    border-color: var(--accent-solid) !important;
}

.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    background: var(--accent-solid-hover) !important;
    color: #ffffff !important;
}

.stButton > button:disabled,
.stDownloadButton > button:disabled {
    opacity: 0.5 !important;
}

.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: var(--input-bg) !important;
    color: var(--input-text) !important;
    border-color: var(--border) !important;
}

.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stMultiSelect label,
.stNumberInput label {
    color: var(--text-primary) !important;
}

input::placeholder,
textarea::placeholder {
    color: var(--text-muted) !important;
}

[data-baseweb="select"] * {
    color: var(--input-text) !important;
}

[data-testid="stFileUploader"] {
    background: transparent !important;
}

[data-testid="stFileUploader"] * {
    color: var(--text-primary) !important;
}

/* The actual visible drag-and-drop box is a separate inner element,
   not just the outer stFileUploader wrapper above -- it needs its
   own background/border or it keeps Streamlit's own default styling
   regardless of the current theme. */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-secondary) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: var(--text-primary) !important;
}

/* "200MB per file - WAV, MP3, ..." helper text: readable but
   secondary, matching the rest of the app's metadata style. */
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--text-secondary) !important;
}

/* "Browse files" button inside the dropzone. */
[data-testid="stFileUploaderDropzone"] button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Uploaded file row (filename, size, remove icon). */
[data-testid="stFileUploaderFile"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 12px 15px;
}

[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
}

button[data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
}

[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] * {
    color: var(--text-primary) !important;
}

[data-testid="stAlert"] {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

[data-testid="stAlert"] * {
    color: var(--text-primary) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
}

pre, code {
    background: var(--code-bg) !important;
    color: var(--text-primary) !important;
}

/* Focus visibility (accessibility) */
.stButton > button:focus-visible,
[data-baseweb="select"] > div:focus-within,
input:focus-visible,
textarea:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 1px;
}

/* ------------------------------------------------------------
   Responsive
   ------------------------------------------------------------ */

@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .page-title { font-size: 25px; }
    .topbar { flex-direction: column; align-items: flex-start; }
}

</style>
""",
    unsafe_allow_html=True,
)



# ============================================================
# HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def _load_json_cached(path_str, _mtime):
    # _mtime is only part of the cache key (so a re-processed file
    # invalidates the cache automatically); it isn't used directly.
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json(path):
    """Reads and parses a JSON file, cached per (path, modification
    time) so repeated reruns -- e.g. every theme toggle click -- don't
    re-read and re-parse every output file from disk each time. The
    mtime key means a freshly re-processed meeting is picked up
    immediately, not served stale."""

    try:
        if not path.exists():
            return None

        return _load_json_cached(str(path), path.stat().st_mtime)

    except Exception:
        return None


def format_time(seconds):
    try:
        seconds = float(seconds)
    except Exception:
        return "00:00"

    minutes = int(seconds // 60)
    secs = int(seconds % 60)

    return f"{minutes:02d}:{secs:02d}"


def format_summary_sentences(summary_text):
    """Split an extractive/abstractive summary into sentences and
    capitalize each one -- the underlying pipeline outputs a single
    lowercase run-on string, which reads poorly as one unbroken block."""

    if not summary_text:
        return []

    import re as _re

    raw_sentences = _re.split(r"(?<=[.!?])\s+", summary_text.strip())

    sentences = []

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence[0].upper() + sentence[1:])

    return sentences


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def get_transcript():
    data = load_json(SPEAKER_TRANSCRIPT)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if isinstance(data.get("segments"), list):
            return data["segments"]

        if isinstance(data.get("transcript"), list):
            return data["transcript"]

    return []


def get_summary_text():
    data = load_json(SUMMARY)

    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        for key in [
            "summary",
            "final_summary",
            "text",
            "generated_summary",
        ]:
            if isinstance(data.get(key), str):
                return data[key]

    return "No summary available yet."


def get_speaker_name(item):
    return (
        item.get("speaker")
        or item.get("speaker_label")
        or item.get("speaker_id")
        or "Unknown Speaker"
    )


def get_segment_text(item):
    return (
        item.get("text")
        or item.get("transcript")
        or ""
    ).strip()


def get_speaker_statistics(transcript):
    stats = {}

    for item in transcript:
        speaker = get_speaker_name(item)

        start = safe_float(item.get("start"))
        end = safe_float(item.get("end"))

        duration = max(0, end - start)

        if speaker not in stats:
            stats[speaker] = 0

        stats[speaker] += duration

    return stats


def get_duration(transcript):
    if not transcript:
        return 0

    ends = [
        safe_float(item.get("end"))
        for item in transcript
        if item.get("end") is not None
    ]

    return max(ends) if ends else 0


def _report_sort_key(report_id):
    """Sorts report_ids newest-first. IDs are "<source>_<YYYYMMDD_HHMMSS>"
    (see build_report's callers) -- stripping the source prefix leaves a
    string that already sorts correctly by recency."""

    for prefix in ("live_", "recorded_"):
        if report_id.startswith(prefix):
            return report_id[len(prefix):]

    return report_id


def _go_to_page(page_name, group="page"):
    """Programmatic nav -- e.g. Home's CTAs and a report's 'Open'
    button. Streamlit forbids writing to a widget's own session_state
    key once that widget has been instantiated in the current run (the
    sidebar radios render before this is ever called), so this stashes
    the target in a separate pending key that gets applied just before
    the sidebar radios are (re-)created on the rerun this triggers."""

    if group == "analytics":
        st.session_state["_pending_nav_analytics"] = page_name
    else:
        st.session_state["_pending_nav_page"] = page_name

    st.session_state["_pending_nav_group"] = group
    st.rerun()


def list_recent_meetings():
    """Every report, newest first, skipping any that fail to load
    (e.g. an older/malformed report shouldn't take down Home)."""

    from meeting_report import list_report_ids, load_report

    reports = []

    for report_id in list_report_ids():

        try:
            reports.append(load_report(report_id))
        except Exception:
            continue

    reports.sort(
        key=lambda r: _report_sort_key(r.meeting_id), reverse=True
    )

    return reports


def _report_mtime(report_id):
    path = OUTPUT_DIR / "reports" / report_id / "report.json"
    return path.stat().st_mtime if path.exists() else 0


@st.cache_data(show_spinner="Computing keywords & topics...")
def compute_report_keywords_topics(report_id, _mtime):
    """Per-report TF-IDF keywords + LDA topics, computed on-demand from
    that report's own transcript (not the single global outputs/nlp/*
    files) -- so switching between meetings in history shows each
    meeting's own keywords/topics. Cached per (report_id, file mtime),
    so this only re-runs when the report actually changes (e.g. a
    speaker rename), not on every rerun/widget click."""

    from meeting_report import load_report
    from nlp_preprocessing import build_processed_segments
    from tfidf_keywords import (
        build_documents as build_tfidf_documents,
        compute_tfidf_keywords,
    )
    from topic_modeling import (
        build_documents as build_topic_documents,
        compute_topics,
    )

    report = load_report(report_id)
    processed = build_processed_segments(report.transcript)

    tfidf_docs, tfidf_meta = build_tfidf_documents(processed)
    topic_docs, topic_meta = build_topic_documents(processed)

    return (
        compute_tfidf_keywords(tfidf_docs, tfidf_meta),
        compute_topics(topic_docs, topic_meta),
    )


# ============================================================
# PIPELINE
# ============================================================

PIPELINE_STEPS = [
    ("Audio preprocessing", "src/audio_preprocessing.py"),
    ("Voice activity detection", "src/vad.py"),
    ("Whisper transcription", "src/transcription.py"),
    ("Speaker diarization", "src/diarization.py"),
    ("Transcript-speaker alignment", "src/merge_transcript.py"),
    ("NLP preprocessing", "src/nlp_preprocessing.py"),
    ("TF-IDF keywords", "src/tfidf_keywords.py"),
    ("Topic modeling", "src/topic_modeling.py"),
    ("Action-item extraction", "src/action_item_extraction.py"),
    ("Transformer summarization", "src/summarization.py"),
]


def run_pipeline(progress_placeholder, log_placeholder):

    logs = []

    # Ensure Homebrew FFmpeg libraries are visible.
    ffmpeg_lib = "/opt/homebrew/opt/ffmpeg/lib"

    env = os.environ.copy()

    current_dyld = env.get("DYLD_LIBRARY_PATH", "")

    if current_dyld:
        env["DYLD_LIBRARY_PATH"] = ffmpeg_lib + ":" + current_dyld
    else:
        env["DYLD_LIBRARY_PATH"] = ffmpeg_lib

    total = len(PIPELINE_STEPS)

    for index, (name, script) in enumerate(PIPELINE_STEPS):

        progress = index / total

        progress_placeholder.progress(
            progress,
            text=f"Processing: {name}"
        )

        logs.append(f"▶ {name}")
        log_placeholder.code("\n".join(logs))

        try:

            result = subprocess.run(
                [sys.executable, script],
                cwd=str(BASE_DIR),
                env=env,
                capture_output=True,
                text=True,
            )

            if result.stdout:
                logs.extend(
                    result.stdout.strip().splitlines()[-8:]
                )

            if result.returncode != 0:

                logs.append(
                    f"ERROR: {name} failed with exit code "
                    f"{result.returncode}"
                )

                if result.stderr:
                    logs.extend(
                        result.stderr.strip().splitlines()[-15:]
                    )

                log_placeholder.code("\n".join(logs))

                return False, "\n".join(logs)

            logs.append(f"OK: {name} completed")
            log_placeholder.code("\n".join(logs))

        except Exception as e:

            logs.append(f"ERROR: {e}")

            log_placeholder.code("\n".join(logs))

            return False, "\n".join(logs)

    progress_placeholder.progress(
        1.0,
        text="Analysis completed successfully"
    )

    logs.append("Complete meeting analysis finished.")

    log_placeholder.code("\n".join(logs))

    return True, "\n".join(logs)





# Apply any pending programmatic nav (from _go_to_page) before the
# sidebar radios below are instantiated -- once instantiated, their
# session_state keys become read-only for the rest of this run.
if "_pending_nav_page" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("_pending_nav_page")
if "_pending_nav_analytics" in st.session_state:
    st.session_state["nav_analytics"] = st.session_state.pop("_pending_nav_analytics")
if "_pending_nav_group" in st.session_state:
    st.session_state["active_nav_group"] = st.session_state.pop("_pending_nav_group")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">∞</div>
            <div class="brand-name">MeetSense</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two nav groups share ONE active page below (see
    # "active_page" right before the page-routing chain). Each
    # group's on_change marks itself as the active group, so
    # switching between GENERAL / RESEARCH behaves like a single
    # combined nav instead of two independent ones.
    def _set_active_nav_group(group_name):
        st.session_state.active_nav_group = group_name

    if "active_nav_group" not in st.session_state:
        st.session_state.active_nav_group = "page"

    st.markdown("### GENERAL")

    page = st.radio(
        "",
        [
            "Home",
            "Meeting Results",
            "Live Meeting",
        ],
        label_visibility="collapsed",
        key="nav_page",
        on_change=_set_active_nav_group,
        args=("page",),
    )

    st.markdown("### RESEARCH")
    st.caption("Model evaluation & benchmarking -- not tied to a specific meeting.")

    analytics_page = st.radio(
        "",
        [
            "Evaluation",
            "Model Comparison",
            "Ablation Study",
            "Error Analysis",
            "Speaker Analysis",
        ],
        label_visibility="collapsed",
        key="nav_analytics",
        on_change=_set_active_nav_group,
        args=("analytics",),
    )

    st.markdown("---")

    st.caption("Speaker-Aware Meeting Intelligence")


# ============================================================
# THEME SWITCH
# ============================================================
# The one control that drives st.session_state.dark_mode, the single
# source of truth for every themed element (see "CUSTOM CSS" above).

_, theme_toggle_col = st.columns([8, 2])

with theme_toggle_col:
    st.toggle(
        "Dark mode",
        key="dark_mode",
        help="Switch between light and dark appearance.",
    )


# TOP BAR
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div class="top-title">
            Meeting Intelligence Workspace
        </div>
        <div>
            <span class="status-pill">● System Ready</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD / NEW MEETING
# ============================================================

_force_expand_upload = st.session_state.pop("_expand_upload", False)

with st.expander(
    "＋  New Meeting / Analyze Audio", expanded=_force_expand_upload
):

    st.markdown(
        "### Upload a meeting recording"
    )

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Upload a meeting recording for complete speaker-aware analysis.",
    )

    if uploaded_file:

        st.audio(
            uploaded_file,
            format=uploaded_file.type,
        )

        st.markdown(
            f"**Selected:** `{uploaded_file.name}`"
        )

        if st.button(
            "Analyze Meeting",
            type="primary",
            use_container_width=True,
        ):

            target = AUDIO_DIR / "meeting.wav"

            try:

                with open(target, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(
                    "Audio uploaded successfully."
                )

                progress_placeholder = st.empty()
                log_placeholder = st.empty()

                with st.spinner(
                    "Running complete meeting analysis..."
                ):

                    success, logs = run_pipeline(
                        progress_placeholder,
                        log_placeholder,
                    )

                if success:

                    st.success(
                        "Meeting analysis completed successfully."
                    )

                    try:
                        from meeting_report import build_report

                        meeting_id = (
                            "recorded_" + time.strftime("%Y%m%d_%H%M%S")
                        )
                        build_report(
                            meeting_id,
                            source="recorded",
                            extractor_name="rule_based",
                        )
                        st.session_state["selected_report_id"] = meeting_id
                    except Exception as report_error:
                        st.warning(
                            f"Analysis succeeded but the meeting report "
                            f"could not be built: {report_error}"
                        )

                    _go_to_page("Meeting Results")

                else:

                    st.error(
                        "The pipeline stopped because one of the processing stages failed."
                    )

                    with st.expander(
                        "View processing logs"
                    ):
                        st.code(logs)

            except Exception as e:

                st.error(
                    f"Could not process uploaded audio: {e}"
                )


# ============================================================
# LOAD DATA
# ============================================================
# Only the static Research-section data is loaded globally -- every
# meeting-specific page (Home, Meeting Results, Live Meeting) loads
# its own report_id-scoped data instead of a single hardcoded "current
# meeting" (see list_recent_meetings/load_report/compute_report_
# keywords_topics above).

wer_data = load_json(WER) or {}
der_data = load_json(DER) or {}
der_vbx_data = load_json(DER_VBX) or {}
rouge_data = load_json(ROUGE) or {}
action_eval_data = load_json(ACTION_EVAL) or {}

model_comparison_data = load_json(MODEL_COMPARISON) or {}
ablation_data = load_json(ABLATION_RESULTS) or {}
error_analysis_data = load_json(ERROR_ANALYSIS) or {}
speaker_analysis_data = load_json(SPEAKER_ANALYSIS) or {}


# ============================================================
# ACTIVE PAGE
# ============================================================
# page / analytics_page are two separate st.radio() widgets that
# both exist in the sidebar simultaneously. active_nav_group (set via
# each radio's on_change) records which group the user actually
# interacted with, so exactly one page renders below instead of only
# ever the GENERAL group's selection.

_active_group = st.session_state.get("active_nav_group", "page")

if _active_group == "analytics":
    active_page = analytics_page
else:
    active_page = page


# ============================================================
# HOME
# ============================================================

if active_page == "Home":

    st.markdown(
        '<div class="page-title">MeetSense</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Upload a recording or start a live '
        "meeting — then open any past meeting's results below.</div>",
        unsafe_allow_html=True,
    )

    cta_col1, cta_col2 = st.columns(2)

    with cta_col1:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Upload a Recording</div>
                <p style="font-size:13px;color:var(--text-secondary);">
                    Analyze an existing audio file — transcript, speaker
                    diarization, summary, decisions and action items.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Upload a Recording", use_container_width=True, type="primary"
        ):
            st.session_state["_expand_upload"] = True
            st.rerun()

    with cta_col2:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Start Live Meeting</div>
                <p style="font-size:13px;color:var(--text-secondary);">
                    Capture and transcribe a meeting as it happens, with
                    live speaker tracking and a rolling summary.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Start Live Meeting", use_container_width=True, type="primary"
        ):
            _go_to_page("Live Meeting")

    st.markdown(
        '<div class="section-heading">Recent Meetings</div>',
        unsafe_allow_html=True,
    )

    recent_meetings = list_recent_meetings()

    if not recent_meetings:

        st.info(
            "No meetings yet — upload a recording or start a live "
            "meeting above to get started."
        )

    else:

        for report in recent_meetings:

            source_badge = (
                "Live" if report.source == "live" else "Recorded"
            )

            row_left, row_right = st.columns([5, 1])

            with row_left:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-title">{report.title or report.meeting_id}</div>
                        <div class="card-subtitle">
                            {report.date}
                            &nbsp; • &nbsp;
                            {format_time(report.duration_sec)}
                            &nbsp; • &nbsp;
                            {len(report.speakers)} speakers
                            &nbsp; • &nbsp;
                            {source_badge}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with row_right:
                if st.button(
                    "Open",
                    key=f"open_report_{report.meeting_id}",
                    use_container_width=True,
                ):
                    st.session_state["selected_report_id"] = report.meeting_id
                    _go_to_page("Meeting Results")


# ============================================================
# LIVE MEETING
# ============================================================

elif active_page == "Live Meeting":

    st.markdown(
        '<div class="page-title">Live Meeting</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Incremental transcription, diarization, '
        "and summarization while the meeting is still running.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                Live Meeting Mode
            </div>
            <p style="font-size:14px;line-height:1.7;color:var(--text-secondary);">
                Audio is captured in rolling 20s windows (3s overlap) and run
                through VAD → Whisper → diarization → alignment → NLP →
                summarization as it arrives. Diarization re-clusters a capped,
                rolling buffer rather than each tiny window in isolation, to
                keep speaker labels consistent -- see PROJECT_REPORT for why.
                Processing speed is <b>not guaranteed to be real-time</b> on
                this hardware; the status strip below reports measured lag,
                not an assumption.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Controller lifecycle: NOT constructed here. Constructing it
    # imports live_controller -> live_pipeline -> vad/transcription/
    # diarization/summarization -> torch/whisper/pyannote/
    # transformers, and loads every model. That only happens inside
    # the Start button handler below, the first time it's clicked --
    # never just from opening this tab. Once constructed, it's
    # cached in st.session_state, so later reruns (every widget
    # interaction) never reconstruct it or reload any model.
    # ------------------------------------------------------------

    from live_capture import list_input_devices  # no torch import -- cheap

    controller = st.session_state.get("live_controller")

    if controller is None and st.session_state.get(
        "live_controller_load_error"
    ):

        st.error(
            "Live-mode models failed to load on the last attempt. "
            "You can try Start again below."
        )

        st.caption(
            f"Details: {st.session_state['live_controller_load_error']}"
        )

    status = (
        controller.get_status()
        if controller is not None
        else {"running": False, "chunk_count": 0}
    )

    # --------------------------------------------------------
    # Controls: source selector + Start/Stop/Reset/Save. Always
    # rendered (even before a controller exists) so the page itself
    # never waits on model loading -- only clicking Start does.
    # --------------------------------------------------------

    source_col, device_col, speaker_count_col = st.columns([1, 2, 1])

    with source_col:

        source_mode = st.radio(
            "Audio source",
            ["Simulated (replay file)", "Microphone"],
            index=0,
            disabled=status["running"],
            help=(
                "Simulated mode replays a saved recording at 1x "
                "speed through the same live pipeline -- the "
                "reliable fallback if microphone access fails "
                "during a demo."
            ),
        )

    with speaker_count_col:

        expected_speaker_count_input = st.number_input(
            "Speakers (if known)",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            disabled=status["running"],
            help=(
                "If you know how many people are speaking, set it "
                "here. Live speaker tracking otherwise invents a new "
                "speaker label whenever a voice doesn't clearly match "
                "an existing one, which fragments one real person "
                "into several labels over a long session. 0 means "
                "unknown -- falls back to the default cap of 8."
            ),
        )

    simulated_file_path = None
    mic_device_index = None

    with device_col:

        if source_mode == "Simulated (replay file)":

            candidate_files = sorted(
                str(p)
                for p in Path("data/audio").glob("*.wav")
            ) + sorted(
                str(p)
                for p in Path("data/test_audio").glob("*.wav")
            )

            if candidate_files:

                simulated_file_path = st.selectbox(
                    "File to replay",
                    candidate_files,
                    disabled=status["running"],
                )

            else:

                st.warning("No .wav files found under data/.")

        else:

            try:
                mic_devices = list_input_devices()
            except Exception:
                mic_devices = []

            if mic_devices:
                selected_device = st.selectbox(
                    "Input device",
                    mic_devices,
                    format_func=lambda d: d["name"],
                    disabled=status["running"],
                    help=(
                        "Which microphone to capture from. Defaults "
                        "to the system's own default input device."
                    ),
                )
                mic_device_index = selected_device["index"]
            else:
                mic_device_index = None
                st.warning(
                    "No microphone input device was detected. "
                    "Use Simulated mode instead."
                )

    start_col, stop_col, reset_col, save_col = st.columns(4)

    with start_col:

        if st.button(
            "▶ Start",
            disabled=status["running"],
            use_container_width=True,
        ):

            try:

                if controller is None:

                    with st.spinner(
                        "Loading live-mode models (VAD, Whisper, "
                        "diarization, summarizer) -- first time "
                        "this session, can take a while under "
                        "memory pressure..."
                    ):

                        from live_controller import LiveSessionController

                        controller = LiveSessionController()

                        st.session_state.live_controller = controller
                        st.session_state.pop(
                            "live_controller_load_error", None
                        )

                _expected_speakers = expected_speaker_count_input or None

                if source_mode == "Microphone":
                    controller.start(
                        mode="microphone",
                        device=mic_device_index,
                        expected_speaker_count=_expected_speakers,
                    )
                else:
                    controller.start(
                        mode="simulated",
                        simulated_file_path=simulated_file_path,
                        expected_speaker_count=_expected_speakers,
                    )

                st.rerun()

            except Exception as error:

                st.session_state["live_start_error"] = str(error)

                if "live_controller" not in st.session_state:
                    st.session_state["live_controller_load_error"] = str(
                        error
                    )

    with stop_col:

        if st.button(
            "■ Stop",
            disabled=not status["running"],
            use_container_width=True,
        ):

            with st.spinner(
                "Stopping... if a chunk is mid-processing this "
                "waits for it to finish (seen up to ~160s during "
                "testing) -- a running model call cannot be "
                "safely interrupted."
            ):
                controller.stop()

            # Stop finalizes the session into a Meeting Report
            # (source="live"), reusing the session's own accumulated
            # transcript/summaries/topics -- no re-reading batch files.
            if controller.session.aligned_transcript:

                from meeting_report import build_report
                import time as _time

                live_meeting_id = (
                    "live_" + _time.strftime("%Y%m%d_%H%M%S")
                )

                _live_extractor_choice = st.session_state.get(
                    "extractor_choice", "rule_based"
                )

                try:

                    _spinner_ctx = (
                        st.spinner("Generating takeaways...")
                        if _live_extractor_choice == "llm"
                        else contextlib.nullcontext()
                    )

                    with _spinner_ctx:

                        live_report = build_report(
                            live_meeting_id,
                            transcript_segments=list(
                                controller.session.aligned_transcript
                            ),
                            chunk_summaries=list(
                                controller.session.chunk_summaries
                            ),
                            final_summary=controller.session.whole_summary,
                            topics=controller.session.topics_result.get(
                                "topics", []
                            ),
                            source="live",
                            extractor_name=_live_extractor_choice,
                            # Models are already loaded in this live
                            # session; only unload before an LLM call.
                            pre_extract_hook=controller.session.unload_models,
                        )

                    st.session_state["live_report_id"] = live_meeting_id

                    if live_report.extractor_used != _live_extractor_choice:
                        st.session_state["live_report_error"] = (
                            live_report.extractor_stats.get(
                                "fallback_notice",
                                f"Fell back to "
                                f"'{live_report.extractor_used}'.",
                            )
                        )

                except Exception as error:
                    st.session_state["live_report_error"] = str(error)

            st.rerun()

    with reset_col:

        if st.button(
            "⟲ Reset",
            disabled=status["running"] or controller is None,
            use_container_width=True,
        ):

            controller.reset()
            st.rerun()

    with save_col:

        if st.button(
            "Save session",
            disabled=(
                controller is None
                or status["running"]
                or status["chunk_count"] == 0
            ),
            use_container_width=True,
        ):

            saved_paths = controller.save_session()
            st.success(f"Saved to {saved_paths['transcript']} and siblings.")

    if st.session_state.get("live_start_error"):

        st.error(
            "Could not start capture. If this is a microphone "
            "permission or device error, switch to Simulated mode "
            "above and try again."
        )
        st.caption(f"Details: {st.session_state['live_start_error']}")

    if st.session_state.get("live_report_id"):

        _just_finished_id = st.session_state["live_report_id"]

        st.success(
            f"Meeting Report ready: {_just_finished_id} "
            f"(see it on Home or Meeting Results)."
        )

        from meeting_report import load_report, save_report

        try:
            _just_finished_report = load_report(_just_finished_id)
        except FileNotFoundError:
            _just_finished_report = None

        if _just_finished_report is not None:

            rename_col, rename_save_col = st.columns([4, 1])

            with rename_col:
                _live_new_title = st.text_input(
                    "Name this meeting",
                    value=_just_finished_report.title or "",
                    placeholder="Untitled meeting -- name it here",
                    key=f"live_title_input_{_just_finished_id}",
                )

            with rename_save_col:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button(
                    "Save name",
                    key=f"live_save_title_{_just_finished_id}",
                    use_container_width=True,
                ):
                    _just_finished_report.title = (
                        _live_new_title.strip() or None
                    )
                    save_report(_just_finished_report)
                    st.success(
                        f"Saved as \"{_just_finished_report.title}\"."
                        if _just_finished_report.title
                        else "Meeting name cleared."
                    )

    if st.session_state.get("live_report_error"):

        st.warning(
            f"Could not build the live Meeting Report: "
            f"{st.session_state['live_report_error']}"
        )

    st.divider()

    if controller is not None:

        # --------------------------------------------------------
        # Live-updating display. This fragment ONLY reads
        # controller.get_status() and renders -- it never calls
        # VAD/ASR/diarization. Re-runs on its own every ~2s without
        # re-running the whole page script.
        # --------------------------------------------------------

        @st.fragment(run_every="2s")
        def render_live_status():

            live_status = controller.get_status()

            # ---- recording indicator ----
            # Always shown while running, regardless of lag state --
            # the lag warning/pill further down is about processing
            # speed, not about whether capture itself is active.

            if live_status["running"]:

                st.markdown(
                    """
                    <div style="
                        display:flex;align-items:center;gap:10px;
                        margin-bottom:12px;
                    ">
                        <span style="
                            width:12px;height:12px;border-radius:50%;
                            background:#e03131;display:inline-block;
                            animation: meetsense-rec-pulse 1.5s infinite;
                        "></span>
                        <span style="font-weight:700;color:#e03131;">
                            Recording
                        </span>
                    </div>
                    <style>
                    @keyframes meetsense-rec-pulse {
                        0% { opacity: 1; }
                        50% { opacity: 0.35; }
                        100% { opacity: 1; }
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

            # ---- status strip ----

            status_cols = st.columns(5)

            status_cols[0].metric(
                "Chunks processed", live_status["chunk_count"]
            )

            status_cols[1].metric(
                "Audio captured",
                format_time(live_status["captured_audio_seconds"]),
            )

            status_cols[2].metric(
                "Audio processed",
                format_time(live_status["processed_audio_seconds"]),
            )

            status_cols[3].metric(
                "Dropped chunks", live_status["dropped_chunk_count"]
            )

            last_log = live_status.get("last_chunk_log") or {}

            status_cols[4].metric(
                "Last chunk total",
                f"{last_log.get('total_ms', 0):.0f} ms",
            )

            if live_status["lag_seconds"] > 5:

                st.warning(
                    f"LAGGING: processing is behind capture by "
                    f"{live_status['lag_seconds']:.1f}s. This is "
                    f"measured, not estimated."
                )

            elif live_status["running"]:

                st.markdown(
                    '<span class="status-pill">● Live -- keeping up '
                    "(lag "
                    f'{live_status["lag_seconds"]:.1f}s)</span>',
                    unsafe_allow_html=True,
                )

            if live_status["stopping"]:
                st.info("Stopping -- waiting for the current chunk to finish...")

            if live_status["last_error"]:
                st.error(f"Pipeline error on last chunk: {live_status['last_error']}")

            if last_log:

                with st.expander("Last chunk -- per-stage timing (ms)"):

                    st.json(
                        {
                            key: value
                            for key, value in last_log.items()
                            if key.endswith("_ms") or key.endswith("_error")
                        }
                    )

            # ---- tabbed meeting content (matches Meeting Results) ----

            live_tab_transcript, live_tab_decisions, live_tab_keywords, live_tab_summary = (
                st.tabs(
                    [
                        "Transcript",
                        "Decisions & Action Items",
                        "Keywords & Topics",
                        "Summary",
                    ]
                )
            )

            with live_tab_transcript:

                st.markdown(
                    '<div class="section-heading">Live Transcript '
                    "(newest first)</div>",
                    unsafe_allow_html=True,
                )

                live_transcript = live_status["transcript"]

                if live_transcript:

                    for segment in reversed(live_transcript[-50:]):

                        st.markdown(
                            f"""
                            <div class="transcript-item">
                                <div class="transcript-header">
                                    <span class="transcript-speaker">
                                         {segment["speaker"]}
                                    </span>
                                    <span class="transcript-time">
                                        {format_time(segment["start"])}
                                    </span>
                                    <span class="transcript-time">
                                        {"speakers refined" if segment.get("speakers_refined") else "provisional speaker"}
                                    </span>
                                </div>
                                <div class="transcript-text">
                                    {segment["text"]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                else:

                    st.caption("No speech processed yet.")

            with live_tab_decisions:

                items_col, decisions_col = st.columns(2)

                with items_col:

                    st.markdown(
                        '<div class="section-heading">Action Items</div>',
                        unsafe_allow_html=True,
                    )

                    if live_status["action_items"]:

                        for item in reversed(live_status["action_items"]):

                            deadline_html = (
                                f'<div class="card-subtitle">Deadline: {item["deadline"]}</div>'
                                if item.get("deadline")
                                else ""
                            )

                            st.markdown(
                                f"""
                                <div class="action-item">
                                    <div class="action-title">□ {item['action']}</div>
                                    <div class="card-subtitle">{item['speaker']}</div>
                                    {deadline_html}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    else:
                        st.caption("None detected yet.")

                with decisions_col:

                    st.markdown(
                        '<div class="section-heading">Decisions</div>',
                        unsafe_allow_html=True,
                    )

                    if live_status["decisions"]:

                        for item in reversed(live_status["decisions"]):

                            st.markdown(
                                f"""
                                <div class="action-item">
                                    <div class="action-title">{item['decision']}</div>
                                    <div class="card-subtitle">{item['speaker']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    else:
                        st.caption("None detected yet.")

            with live_tab_keywords:

                st.markdown(
                    '<div class="section-heading">Top Keywords & Topics</div>',
                    unsafe_allow_html=True,
                )

                keywords = live_status["keywords"].get("top_keywords", [])
                topics = live_status["topics"].get("topics", [])

                if keywords or topics:

                    keyword_col, topic_col = st.columns(2)

                    with keyword_col:

                        if keywords:
                            st.markdown(
                                ", ".join(
                                    k["term"] for k in keywords[:15]
                                )
                            )
                        else:
                            st.caption("No keywords yet.")

                    with topic_col:

                        if topics:
                            for topic in topics:
                                words = ", ".join(
                                    w["word"] for w in topic["top_words"][:6]
                                )
                                st.markdown(f"**Topic {topic['topic_id']}:** {words}")
                        else:
                            st.caption("No topics yet.")

                else:

                    st.caption(
                        "Keywords/topics refit every "
                        f"{controller.session.recompute_every_n_chunks} chunks "
                        "over the growing transcript -- none yet."
                    )

            with live_tab_summary:

                st.markdown(
                    '<div class="section-heading">Rolling Meeting Summary</div>',
                    unsafe_allow_html=True,
                )

                if live_status["whole_summary"]:

                    live_summary_html = "".join(
                        f"<p style='margin:0 0 10px 0;'>{sentence}</p>"
                        for sentence in format_summary_sentences(
                            live_status["whole_summary"]
                        )
                    )

                    st.markdown(
                        f"""
                        <div class="summary-box">{live_summary_html}</div>
                        <div class="card-subtitle">
                            As of chunk {live_status["chunk_count"]} /
                            {format_time(live_status["processed_audio_seconds"])}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:

                    st.caption(
                        "Whole-meeting summary regenerates every "
                        f"{controller.session.recompute_every_n_chunks} chunks "
                        "-- none yet."
                    )

                if live_status["chunk_summaries"]:

                    with st.expander(
                        f"Chunk summaries so far "
                        f"({len(live_status['chunk_summaries'])})"
                    ):

                        for index, chunk_summary in enumerate(
                            live_status["chunk_summaries"], start=1
                        ):
                            st.markdown(f"**Chunk {index}:** {chunk_summary}")

        render_live_status()


# ============================================================
# TRANSCRIPT PAGE
# ============================================================

elif active_page == "Meeting Results":

    st.markdown(
        '<div class="page-title">Meeting Results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Transcript, speakers, summary, '
        "decisions, action items, keywords and the full report -- "
        "for one meeting at a time.</div>",
        unsafe_allow_html=True,
    )

    from meeting_report import (
        build_report,
        load_report,
        save_report,
        list_report_ids,
        apply_speaker_renaming,
        render_markdown,
        render_html,
        available_extractors,
    )

    existing_report_ids = sorted(
        list_report_ids(), key=_report_sort_key, reverse=True
    )

    if not existing_report_ids:

        st.info(
            "No analyzed meetings yet. Upload a recording or start a "
            "live meeting from Home."
        )

    else:

        preferred_report_id = st.session_state.get("selected_report_id")

        default_index = (
            existing_report_ids.index(preferred_report_id)
            if preferred_report_id in existing_report_ids
            else 0
        )

        select_col, extractor_col = st.columns([2, 1])

        with select_col:
            report_id = st.selectbox(
                "Meeting", existing_report_ids, index=default_index
            )

        with extractor_col:
            # key="extractor_choice" persists this selection in
            # session_state so the Live Meeting Stop handler (a
            # different page) can reuse the same preference for
            # finalizing a live report with "the selected extractor".
            extractor_choice = st.selectbox(
                "Extractor",
                available_extractors(),
                index=0,
                key="extractor_choice",
            )

        st.session_state["selected_report_id"] = report_id

        try:
            meeting_report = load_report(report_id)
        except FileNotFoundError:
            meeting_report = None

        if meeting_report is None:

            st.info("No report found for this meeting.")

        else:

            name_col, name_save_col = st.columns([4, 1])

            with name_col:
                new_title = st.text_input(
                    "Meeting name",
                    value=meeting_report.title or "",
                    placeholder="Untitled meeting -- name it here",
                    key=f"title_input_{report_id}",
                    label_visibility="collapsed",
                )

            with name_save_col:
                if st.button(
                    "Save name",
                    key=f"save_title_{report_id}",
                    use_container_width=True,
                ):
                    meeting_report.title = new_title.strip() or None
                    save_report(meeting_report)
                    st.success(
                        f'Saved as "{meeting_report.title}".'
                        if meeting_report.title
                        else "Meeting name cleared."
                    )

            report_transcript = meeting_report.transcript
            report_duration = meeting_report.duration_sec
            report_speaker_count = len(meeting_report.speakers)
            report_segment_count = len(report_transcript)

            (
                tab_transcript,
                tab_speakers,
                tab_summary,
                tab_decisions,
                tab_keywords,
                tab_report,
            ) = st.tabs(
                [
                    "Transcript",
                    "Speakers",
                    "Summary",
                    "Decisions & Action Items",
                    "Keywords & Topics",
                    "Report",
                ]
            )

            # --------------------------------------------------
            # TRANSCRIPT TAB
            # --------------------------------------------------
            with tab_transcript:

                search = st.text_input(
                    "Search transcript",
                    placeholder="Search words, sentences or speakers...",
                    key=f"transcript_search_{report_id}",
                )

                filtered = report_transcript

                if search:

                    query = search.lower()

                    filtered = [
                        item
                        for item in report_transcript
                        if query in get_segment_text(item).lower()
                        or query in get_speaker_name(item).lower()
                    ]

                st.caption(f"{len(filtered)} transcript segments")

                segment_blocks = []

                for item in filtered:

                    speaker = get_speaker_name(item)
                    text = get_segment_text(item)

                    segment_blocks.append(
                        f"""
                        <div class="transcript-item">
                            <div class="transcript-header">
                                <span class="transcript-speaker">
                                    {speaker}
                                </span>
                                <span class="transcript-time">
                                    {format_time(item.get("start",0))}
                                    —
                                    {format_time(item.get("end",0))}
                                </span>
                            </div>

                            <div class="transcript-text">
                                {text}
                            </div>
                        </div>
                        """
                    )

                if segment_blocks:
                    st.markdown(
                        "".join(segment_blocks), unsafe_allow_html=True
                    )
                else:
                    st.info("No transcript segments for this meeting.")

            # --------------------------------------------------
            # SPEAKERS TAB
            # --------------------------------------------------
            with tab_speakers:

                if meeting_report.speakers:

                    import pandas as pd

                    chart_df = pd.DataFrame(
                        {
                            "speaker": [
                                s.name for s in meeting_report.speakers
                            ],
                            "seconds": [
                                s.speaking_time_sec
                                for s in meeting_report.speakers
                            ],
                        }
                    ).set_index("speaker")

                    st.bar_chart(chart_df)

                try:
                    tfidf_result, topic_result = (
                        compute_report_keywords_topics(
                            report_id, _report_mtime(report_id)
                        )
                    )
                except Exception:
                    tfidf_result, topic_result = {}, {}

                try:
                    from speaker_analysis import compute_speaker_contributions

                    normalized_action_data = {
                        "decisions": [
                            {
                                "decision": item.text,
                                "speaker": item.evidence.speaker,
                                "segment_id": None,
                            }
                            for item in meeting_report.decisions
                        ],
                        "action_items": [
                            {
                                "action": item.text,
                                "speaker": item.evidence.speaker,
                                "segment_id": None,
                            }
                            for item in meeting_report.action_items
                        ],
                    }

                    speaker_contributions, _ = compute_speaker_contributions(
                        report_transcript,
                        tfidf_result,
                        topic_result,
                        normalized_action_data,
                    )
                except Exception:
                    speaker_contributions = []

                if not speaker_contributions:

                    st.info("No per-speaker breakdown for this meeting.")

                else:

                    st.caption(
                        "All figures below are objective counts/"
                        "measurements (speaking time, participation "
                        "percentage, utterance/question counts, topic "
                        "and keyword association, decisions/action "
                        "items). Not a judgment of performance."
                    )

                    for speaker in speaker_contributions:

                        topic_labels = []

                        for t in speaker.get("topics", []):
                            label = (
                                ", ".join(t["top_words"])
                                or f"Topic {t['topic_id']}"
                            )
                            topic_labels.append(label)

                        topics_html = "".join(
                            f'<span class="tag">{label}</span>'
                            for label in topic_labels
                        ) or (
                            '<span class="card-subtitle">'
                            "No dominant topics.</span>"
                        )

                        keywords_html = "".join(
                            f'<span class="tag">{kw.get("term")}</span>'
                            for kw in speaker.get("keywords", [])
                        ) or (
                            '<span class="card-subtitle">'
                            "No keywords.</span>"
                        )

                        st.markdown(
                            f"""
                            <div class="card">
                                <div style="display:flex;justify-content:space-between;align-items:baseline;">
                                    <div class="card-title">{speaker['display_name']}</div>
                                    <div style="font-size:20px;font-weight:800;color:var(--accent);">
                                        {speaker['participation_percentage']:.1f}%
                                    </div>
                                </div>
                                <div style="
                                    margin-top:10px;
                                    height:8px;
                                    background:var(--border);
                                    border-radius:10px;
                                ">
                                    <div style="
                                        width:{min(speaker['participation_percentage'], 100):.1f}%;
                                        height:8px;
                                        background:var(--accent);
                                        border-radius:10px;
                                    "></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        m1, m2, m3, m4, m5 = st.columns(5)

                        with m1:
                            st.markdown(
                                f"""<div class="metric">
                                    <div class="metric-name">Speaking time</div>
                                    <div class="metric-value">{speaker['speaking_time_formatted']}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                        with m2:
                            st.markdown(
                                f"""<div class="metric">
                                    <div class="metric-name">Utterances</div>
                                    <div class="metric-value">{speaker['utterance_count']}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                        with m3:
                            st.markdown(
                                f"""<div class="metric">
                                    <div class="metric-name">Questions</div>
                                    <div class="metric-value">{speaker['question_count']}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                        with m4:
                            st.markdown(
                                f"""<div class="metric">
                                    <div class="metric-name">Decisions</div>
                                    <div class="metric-value">{speaker['decision_count']}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                        with m5:
                            st.markdown(
                                f"""<div class="metric">
                                    <div class="metric-name">Action items</div>
                                    <div class="metric-value">{speaker['action_item_count']}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                        st.markdown(
                            f'<div class="card-subtitle" style="margin-top:10px;">Topics</div>{topics_html}',
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f'<div class="card-subtitle" style="margin-top:10px;">Keywords</div>{keywords_html}',
                            unsafe_allow_html=True,
                        )

                        contributions = speaker.get(
                            "notable_contributions", []
                        )

                        if contributions:

                            st.markdown(
                                '<div class="card-subtitle" style="margin-top:10px;">'
                                "Notable contributions (own extracted "
                                "decisions/action items)</div>",
                                unsafe_allow_html=True,
                            )

                            for item in contributions:

                                label = (
                                    "Decision"
                                    if item["type"] == "decision"
                                    else "Action item"
                                )

                                st.markdown(
                                    f'<div style="font-size:13px;margin-top:4px;">{label}: {item["text"]}</div>',
                                    unsafe_allow_html=True,
                                )

                        st.markdown(
                            "<div style='height:20px'></div>",
                            unsafe_allow_html=True,
                        )

            # --------------------------------------------------
            # SUMMARY TAB
            # --------------------------------------------------
            with tab_summary:

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric("Duration", format_time(report_duration))

                with c2:
                    st.metric("Speakers", report_speaker_count)

                with c3:
                    st.metric("Segments", report_segment_count)

                st.markdown(
                    '<div class="section-heading">Summary</div>',
                    unsafe_allow_html=True,
                )

                summary_sentences = format_summary_sentences(
                    meeting_report.summary
                )

                if summary_sentences:
                    summary_html = "".join(
                        f"<p style='margin:0 0 10px 0;'>{sentence}</p>"
                        for sentence in summary_sentences
                    )
                else:
                    summary_html = "No summary available yet."

                st.markdown(
                    f'<div class="summary-box">{summary_html}</div>',
                    unsafe_allow_html=True,
                )

            # --------------------------------------------------
            # DECISIONS & ACTION ITEMS TAB
            # --------------------------------------------------
            with tab_decisions:

                any_section = False

                for section_name, section in [
                    ("Decisions", meeting_report.decisions),
                    ("Action Items", meeting_report.action_items),
                    ("Deadlines", meeting_report.deadlines),
                    ("Open Questions", meeting_report.open_questions),
                ]:

                    if section:

                        any_section = True

                        st.markdown(
                            f'<div class="section-heading">{section_name}</div>',
                            unsafe_allow_html=True,
                        )

                        for item in section:

                            due = (
                                f" · due {item.due_date}"
                                if item.due_date
                                else ""
                            )

                            st.markdown(f"**{item.text}**{due}")

                            with st.expander(
                                f"Evidence — {item.evidence.speaker} @ "
                                f"{format_time(item.evidence.timestamp)}"
                            ):
                                st.markdown(f"> {item.evidence.quote}")

                if not any_section:
                    st.info(
                        "No decisions, action items, deadlines or open "
                        "questions were extracted for this meeting."
                    )

            # --------------------------------------------------
            # KEYWORDS & TOPICS TAB
            # --------------------------------------------------
            with tab_keywords:

                st.markdown(
                    '<div class="section-heading">TF-IDF Keywords</div>',
                    unsafe_allow_html=True,
                )

                tab_keywords_list = (
                    tfidf_result.get("top_keywords")
                    if isinstance(tfidf_result, dict)
                    else None
                ) or []

                if tab_keywords_list:

                    for item in tab_keywords_list:

                        word = item.get("term", "")
                        score = item.get("score")

                        if score is not None:

                            st.markdown(
                                f"""
                                <div class="speaker-row">
                                    <div style="display:flex;justify-content:space-between;">
                                        <span class="speaker-name">{word}</span>
                                        <span class="speaker-time">{safe_float(score):.4f}</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        elif word:

                            st.markdown(
                                f'<span class="tag">{word}</span>',
                                unsafe_allow_html=True,
                            )

                else:

                    st.info(
                        "No TF-IDF keyword data available for this "
                        "meeting."
                    )

                st.markdown(
                    '<div class="section-heading">LDA Topics</div>',
                    unsafe_allow_html=True,
                )

                tab_topics_list = (
                    topic_result.get("topics")
                    if isinstance(topic_result, dict)
                    else None
                ) or []

                if tab_topics_list:

                    for index, topic in enumerate(tab_topics_list, start=1):

                        words = topic.get("top_words", [])

                        st.markdown(
                            f"""
                            <div class="card">
                                <div class="card-title">
                                    Topic {index}
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        for word in words:

                            label = (
                                word.get("word")
                                if isinstance(word, dict)
                                else str(word)
                            )

                            st.markdown(
                                f'<span class="tag">{label}</span>',
                                unsafe_allow_html=True,
                            )

                        st.markdown("</div>", unsafe_allow_html=True)

                else:

                    st.info(
                        "No topic modeling data available for this "
                        "meeting."
                    )

            # --------------------------------------------------
            # REPORT TAB
            # --------------------------------------------------
            with tab_report:

                if st.button(
                    "Rebuild from current batch outputs",
                    key=f"rebuild_report_{report_id}",
                ):

                    _live_controller = st.session_state.get(
                        "live_controller"
                    )

                    _unload_hook = (
                        _live_controller.session.unload_models
                        if _live_controller is not None
                        else None
                    )

                    try:

                        if extractor_choice == "llm":
                            with st.spinner("Generating takeaways..."):
                                built_report = build_report(
                                    report_id,
                                    source="recorded",
                                    extractor_name=extractor_choice,
                                    pre_extract_hook=_unload_hook,
                                )
                        else:
                            built_report = build_report(
                                report_id,
                                source="recorded",
                                extractor_name=extractor_choice,
                            )

                        if built_report.extractor_used != extractor_choice:
                            st.warning(
                                built_report.extractor_stats.get(
                                    "fallback_notice",
                                    f"Fell back to '{built_report.extractor_used}'.",
                                )
                            )
                        else:
                            st.success(
                                f"Rebuilt '{report_id}' using "
                                f"'{extractor_choice}'."
                            )

                        st.rerun()

                    except Exception as error:
                        st.error(f"Could not rebuild report: {error}")

                if meeting_report.source == "live":

                    audio_path = Path(
                        "outputs/live/live_session_audio.wav"
                    )
                    refine_key = f"offline_refine_{report_id}"

                    if st.button(
                        "Refine speakers (offline, slower)",
                        disabled=not audio_path.exists(),
                        key=f"refine_speakers_{report_id}",
                    ):
                        status = {
                            "running": True,
                            "done": False,
                            "error": None,
                            "rerun_requested": False,
                        }
                        st.session_state[refine_key] = status
                        threading.Thread(
                            target=_refine_live_report_speakers,
                            args=(report_id, str(audio_path), status),
                            daemon=True,
                        ).start()

                    if not audio_path.exists():
                        st.caption(
                            "Saved live audio is unavailable for "
                            "offline refinement."
                        )

                    @st.fragment(run_every="2s")
                    def show_speaker_refinement_status():
                        status = st.session_state.get(refine_key)
                        if not status:
                            return
                        if status.get("running"):
                            st.info(
                                "Offline speaker refinement is running "
                                "in the background."
                            )
                        elif status.get("error"):
                            st.error(
                                f"Offline speaker refinement failed: "
                                f"{status['error']}"
                            )
                        elif status.get("done"):
                            st.success(
                                "Speaker labels were refined and the "
                                "report was updated."
                            )
                            if not status.get("rerun_requested"):
                                status["rerun_requested"] = True
                                st.rerun()

                    show_speaker_refinement_status()

                st.markdown(
                    f"**Date:** {meeting_report.date} &nbsp;&nbsp; "
                    f"**Duration:** {format_time(meeting_report.duration_sec)} "
                    f"&nbsp;&nbsp; **Source:** {meeting_report.source}"
                )

                if meeting_report.speakers:

                    with st.expander("Rename speakers"):

                        renaming = {}

                        for speaker in meeting_report.speakers:

                            new_name = st.text_input(
                                f"Rename {speaker.name}",
                                value=speaker.name,
                                key=f"rename_{report_id}_{speaker.name}",
                            )

                            if new_name != speaker.name:
                                renaming[speaker.name] = new_name

                        if renaming and st.button(
                            "Apply renaming", key=f"apply_rename_{report_id}"
                        ):

                            meeting_report = apply_speaker_renaming(
                                meeting_report, renaming
                            )
                            save_report(meeting_report)

                            st.success(
                                "Renamed. Applied to speakers, "
                                "transcript, and all evidence."
                            )
                            st.rerun()

                st.markdown(
                    '<div class="section-heading">Export</div>',
                    unsafe_allow_html=True,
                )

                export_col1, export_col2, export_col3 = st.columns(3)

                with export_col1:
                    st.download_button(
                        "JSON",
                        data=json.dumps(
                            meeting_report.to_rendered_dict(),
                            indent=2,
                            ensure_ascii=False,
                        ),
                        file_name=f"{report_id}_report.json",
                        mime="application/json",
                        key=f"export_json_{report_id}",
                    )

                with export_col2:
                    st.download_button(
                        "Markdown",
                        data=render_markdown(meeting_report),
                        file_name=f"{report_id}_report.md",
                        mime="text/markdown",
                        key=f"export_md_{report_id}",
                    )

                with export_col3:
                    st.download_button(
                        "HTML",
                        data=render_html(meeting_report),
                        file_name=f"{report_id}_report.html",
                        mime="text/html",
                        key=f"export_html_{report_id}",
                    )


# ============================================================
# SPEAKER ANALYSIS
# ============================================================

elif active_page == "Speaker Analysis":

    st.markdown(
        '<div class="page-title">Speaker Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Objective, measurable per-speaker '
        'contribution -- speaking time, topics, keywords, questions, '
        'decisions and action items.</div>',
        unsafe_allow_html=True,
    )

    if not speaker_analysis_data:

        st.info(
            "No speaker analysis data available. Run "
            "src/speaker_analysis.py to generate it."
        )

    else:

        st.caption(
            f"{speaker_analysis_data.get('objective_measurement_only', '')}"
        )

        for speaker in speaker_analysis_data.get("speakers", []):

            topic_labels = []

            for t in speaker.get("topics", []):

                label = ", ".join(t["top_words"]) or f"Topic {t['topic_id']}"
                topic_labels.append(label)

            topics_html = "".join(
                f'<span class="tag">{label}</span>' for label in topic_labels
            ) or "<span class=\"card-subtitle\">No dominant topics.</span>"

            keywords_html = "".join(
                f'<span class="tag">{kw.get("term")}</span>'
                for kw in speaker.get("keywords", [])
            ) or "<span class=\"card-subtitle\">No keywords.</span>"

            st.markdown(
                f"""
                <div class="card">
                    <div style="display:flex;justify-content:space-between;align-items:baseline;">
                        <div class="card-title">{speaker['display_name']}</div>
                        <div style="font-size:20px;font-weight:800;color:var(--accent);">
                            {speaker['participation_percentage']:.1f}%
                        </div>
                    </div>
                    <div style="
                        margin-top:10px;
                        height:8px;
                        background:var(--border);
                        border-radius:10px;
                    ">
                        <div style="
                            width:{min(speaker['participation_percentage'], 100):.1f}%;
                            height:8px;
                            background:var(--accent);
                            border-radius:10px;
                        "></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4, m5 = st.columns(5)

            with m1:
                st.markdown(
                    f"""<div class="metric">
                        <div class="metric-name">Speaking time</div>
                        <div class="metric-value">{speaker['speaking_time_formatted']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with m2:
                st.markdown(
                    f"""<div class="metric">
                        <div class="metric-name">Utterances</div>
                        <div class="metric-value">{speaker['utterance_count']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with m3:
                st.markdown(
                    f"""<div class="metric">
                        <div class="metric-name">Questions</div>
                        <div class="metric-value">{speaker['question_count']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with m4:
                st.markdown(
                    f"""<div class="metric">
                        <div class="metric-name">Decisions</div>
                        <div class="metric-value">{speaker['decision_count']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with m5:
                st.markdown(
                    f"""<div class="metric">
                        <div class="metric-name">Action items</div>
                        <div class="metric-value">{speaker['action_item_count']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="card-subtitle" style="margin-top:10px;">Topics</div>{topics_html}',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="card-subtitle" style="margin-top:10px;">Keywords</div>{keywords_html}',
                unsafe_allow_html=True,
            )

            contributions = speaker.get("notable_contributions", [])

            if contributions:

                st.markdown(
                    '<div class="card-subtitle" style="margin-top:10px;">'
                    'Notable contributions (own extracted decisions/action items)</div>',
                    unsafe_allow_html=True,
                )

                for item in contributions:

                    label = "Decision" if item["type"] == "decision" else "Action item"

                    st.markdown(
                        f'<div style="font-size:13px;margin-top:4px;">{label}: {item["text"]}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        if speaker_analysis_data.get("methodology"):

            with st.expander("Methodology"):

                for key, value in speaker_analysis_data["methodology"].items():
                    st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")


# ============================================================
# EVALUATION
# ============================================================

elif active_page == "Evaluation":

    st.markdown(
        '<div class="page-title">Model Evaluation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Quantitative evaluation of transcription, diarization, summarization and action extraction.</div>',
        unsafe_allow_html=True,
    )

    # ---------------- WER ----------------

    st.markdown(
        '<div class="section-heading">Speech Recognition</div>',
        unsafe_allow_html=True,
    )

    wer_value = safe_float(
        wer_data.get("wer", 0)
        if isinstance(wer_data, dict)
        else 0
    )

    # Diarization Method selector -- PyAnnote is the production backend
    # and is always available; VBx (experimental) only appears once its
    # own DER results exist (i.e. after running
    # `python run_diarization_experiment.py --method vbx`), so the
    # toggle never offers a method with nothing to show.
    diarization_method = "PyAnnote (production)"

    if der_vbx_data:

        diarization_method = st.selectbox(
            "Diarization Method",
            ["PyAnnote (production)", "VBx (experimental)"],
            key="diarization_method_selector",
        )

    if diarization_method.startswith("VBx"):

        der_value = safe_float(der_vbx_data.get("der", 0))

        st.caption(
            "VBx is an experimental, research-comparison-only "
            "diarization backend. It received different upstream "
            "segmentation input than PyAnnote (externally supplied "
            "VAD vs. PyAnnote's own internal segmentation), so this "
            "DER is not a like-for-like clustering-quality comparison "
            "-- see README.md for details."
        )

    else:

        der_value = safe_float(
            der_data.get("der", 0)
            if isinstance(der_data, dict)
            else 0
        )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">Word Error Rate</div>
                <div class="metric-value">
                    {wer_value * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">Diarization Error Rate</div>
                <div class="metric-value">
                    {der_value * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- ROUGE ----------------

    st.markdown(
        '<div class="section-heading">Summarization</div>',
        unsafe_allow_html=True,
    )

    rouge1 = 0
    rouge2 = 0
    rougeL = 0

    if isinstance(rouge_data, dict):

        r1 = rouge_data.get("rouge1", {})
        r2 = rouge_data.get("rouge2", {})
        rl = rouge_data.get("rougeL", {})

        if isinstance(r1, dict):
            rouge1 = safe_float(
                r1.get("f1")
                or r1.get("fmeasure")
                or 0
            )

        if isinstance(r2, dict):
            rouge2 = safe_float(
                r2.get("f1")
                or r2.get("fmeasure")
                or 0
            )

        if isinstance(rl, dict):
            rougeL = safe_float(
                rl.get("f1")
                or rl.get("fmeasure")
                or 0
            )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">ROUGE-1 F1</div>
                <div class="metric-value">
                    {rouge1 * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r2:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">ROUGE-2 F1</div>
                <div class="metric-value">
                    {rouge2 * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r3:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">ROUGE-L F1</div>
                <div class="metric-value">
                    {rougeL * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- ACTION ITEM ----------------

    st.markdown(
        '<div class="section-heading">Action Item Extraction</div>',
        unsafe_allow_html=True,
    )

    action_f1 = 0
    action_precision = 0
    action_recall = 0

    if isinstance(action_eval_data, dict):

        action_f1 = safe_float(
            action_eval_data.get("f1")
            or action_eval_data.get("f1_score")
            or 0
        )

        action_precision = safe_float(
            action_eval_data.get("precision")
            or 0
        )

        action_recall = safe_float(
            action_eval_data.get("recall")
            or 0
        )

    a1, a2, a3 = st.columns(3)

    with a1:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">Precision</div>
                <div class="metric-value">
                    {action_precision * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">Recall</div>
                <div class="metric-value">
                    {action_recall * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a3:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">F1 Score</div>
                <div class="metric-value">
                    {action_f1 * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif active_page == "Model Comparison":

    st.markdown(
        '<div class="page-title">Model Comparison</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Comparing installed model variants on this recording.</div>',
        unsafe_allow_html=True,
    )

    if not model_comparison_data:

        st.info(
            "No model comparison data available. Run "
            "src/model_comparison.py to generate it."
        )

    else:

        for note in model_comparison_data.get("limitations", []):
            st.caption(f"{note}")

        st.markdown(
            '<div class="section-heading">ASR: Whisper base vs small</div>',
            unsafe_allow_html=True,
        )

        asr = model_comparison_data.get("asr_comparison", {})
        asr_models = asr.get("models", {})

        if asr_models:

            asr_cols = st.columns(len(asr_models))

            for col, (model_key, model_info) in zip(asr_cols, asr_models.items()):

                with col:

                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-title">{model_key}</div>
                            <div class="metric">
                                <div class="metric-name">WER</div>
                                <div class="metric-value">
                                    {safe_float(model_info.get("wer_percentage")):.2f}%
                                </div>
                            </div>
                            <div class="metric">
                                <div class="metric-name">Transcribe time</div>
                                <div class="metric-value">
                                    {safe_float(model_info.get("transcribe_seconds")):.1f}s
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            cross_check = asr.get("cross_check_against_wer_results_json", {})

            if cross_check.get("available"):

                st.caption(
                    "Cross-check vs the standalone WER evaluation: "
                    f"{safe_float(cross_check.get('stored_wer_percentage')):.2f}% stored, "
                    f"{safe_float(cross_check.get('fresh_base_wer_percentage')):.2f}% "
                    "freshly re-run base model."
                )

        else:

            st.info("No ASR comparison data available.")

        st.markdown(
            '<div class="section-heading">Summarization: T5-small vs TF-IDF extractive</div>',
            unsafe_allow_html=True,
        )

        summarization_comparison = model_comparison_data.get(
            "summarization_comparison", {}
        )
        sum_models = summarization_comparison.get("models", {})

        if sum_models:

            sum_cols = st.columns(len(sum_models))

            for col, (model_key, model_info) in zip(sum_cols, sum_models.items()):

                with col:

                    rouge1 = model_info.get("rouge", {}).get("rouge1", {})

                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-title">{model_key}</div>
                            <div class="metric">
                                <div class="metric-name">ROUGE-1 F1</div>
                                <div class="metric-value">
                                    {safe_float(rouge1.get("fmeasure")) * 100:.2f}%
                                </div>
                            </div>
                            <div class="metric">
                                <div class="metric-name">Summary length</div>
                                <div class="metric-value">
                                    {model_info.get("summary_word_count", "—")} words
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if summarization_comparison.get("note"):
                st.caption(summarization_comparison["note"])

        else:

            st.info("No summarization comparison data available.")

        st.markdown(
            '<div class="section-heading">Diarization</div>',
            unsafe_allow_html=True,
        )

        diarization_comparison = model_comparison_data.get(
            "diarization_comparison", {}
        )

        st.info(
            diarization_comparison.get(
                "reason",
                "No diarization model comparison available.",
            )
        )


# ============================================================
# ABLATION STUDY
# ============================================================

elif active_page == "Ablation Study":

    st.markdown(
        '<div class="page-title">Ablation Study</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Whisper only → + VAD → + diarization → full pipeline.</div>',
        unsafe_allow_html=True,
    )

    if not ablation_data:

        st.info(
            "No ablation data available. Run src/ablation_study.py "
            "to generate it."
        )

    else:

        for note in ablation_data.get("limitations", []):
            st.caption(f"{note}")

        stage_labels = {
            "whisper_only": "1 · Whisper only",
            "whisper_plus_vad": "2 · + VAD",
            "whisper_vad_diarization": "3 · + Diarization",
            "full_pipeline": "4 · Full pipeline",
        }

        metric_specs = [
            ("WER", "wer_percentage", "%"),
            ("DER", "der_percentage", "%"),
            ("Action F1", "action_item_f1", ""),
            ("ROUGE-1 F1", "rouge1_f1", ""),
        ]

        for stage in ablation_data.get("stages", []):

            stage_key = stage.get("stage", "")
            label = stage_labels.get(stage_key, stage_key)

            st.markdown(
                f'<div class="section-heading">{label}</div>',
                unsafe_allow_html=True,
            )

            st.caption(stage.get("description", ""))

            metrics = stage.get("metrics", {})
            not_applicable = stage.get("not_applicable", {})

            metric_cols = st.columns(len(metric_specs))

            for col, (name, key, suffix) in zip(metric_cols, metric_specs):

                with col:

                    value = metrics.get(key)

                    if value is None:
                        display = "—"
                    elif suffix == "%":
                        display = f"{safe_float(value):.2f}%"
                    else:
                        display = f"{safe_float(value):.4f}"

                    st.markdown(
                        f"""
                        <div class="metric">
                            <div class="metric-name">{name}</div>
                            <div class="metric-value">{display}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if not_applicable:

                reasons = " · ".join(
                    f"{k}: {v}" for k, v in not_applicable.items()
                )

                st.caption(f"Not applicable — {reasons}")

            if stage.get("note"):
                st.caption(stage["note"])


# ============================================================
# ERROR ANALYSIS
# ============================================================

elif active_page == "Error Analysis":

    st.markdown(
        '<div class="page-title">Error Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Structured error findings across ASR, diarization, NLP, action items and summarization.</div>',
        unsafe_allow_html=True,
    )

    if not error_analysis_data:

        st.info(
            "No error analysis data available. Run "
            "src/error_analysis.py to generate it."
        )

    else:

        for note in error_analysis_data.get("limitations", []):
            st.caption(f"{note}")

        # ---------------- ASR ----------------

        asr_errors = error_analysis_data.get("asr", {})

        with st.expander("ASR errors", expanded=True):

            word_errors = asr_errors.get("word_level_errors", {})
            wer_info = asr_errors.get("word_error_rate", {})

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Substitutions</div>
                        <div class="metric-value">{word_errors.get("substitutions", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Insertions</div>
                        <div class="metric-value">{word_errors.get("insertions", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Deletions (missed)</div>
                        <div class="metric-value">{word_errors.get("deletions", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "Punctuation-attributable WER delta: "
                f"{safe_float(wer_info.get('punctuation_attributable_delta_points')):.2f} "
                "percentage points"
            )

            sub_examples = word_errors.get("substitution_examples", [])

            if sub_examples:

                st.markdown("**Substitution examples**")

                for ex in sub_examples[:8]:
                    st.markdown(
                        f'<span class="tag">{ex.get("reference")} → {ex.get("hypothesis")}</span>',
                        unsafe_allow_html=True,
                    )

            filler_counts = asr_errors.get("filler_word_usage", {}).get("counts", {})

            if filler_counts:

                st.markdown("**Filler word usage (reference vs hypothesis)**")

                for term, counts in filler_counts.items():
                    st.caption(
                        f"'{term}' — reference: {counts.get('reference')}, "
                        f"hypothesis: {counts.get('hypothesis')}"
                    )

        # ---------------- Diarization ----------------

        diarization_errors = error_analysis_data.get("diarization", {})

        with st.expander("Diarization errors"):

            confusion = diarization_errors.get("speaker_confusion", {})
            overlap = diarization_errors.get("overlapping_speech", {})
            boundary = diarization_errors.get("boundary_confidence", {})

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Confusion (s)</div>
                        <div class="metric-value">{safe_float(confusion.get("confusion_seconds")):.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Overlapping turn pairs</div>
                        <div class="metric-value">{overlap.get("overlapping_turn_pairs_found", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Low-confidence boundaries</div>
                        <div class="metric-value">{safe_float(boundary.get("segments_below_50pct_overlap_percentage")):.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if boundary.get("methodology"):
                st.caption(boundary["methodology"])

        # ---------------- NLP ----------------

        nlp_errors = error_analysis_data.get("nlp", {})

        with st.expander("NLP errors"):

            ner = nlp_errors.get("ner_false_positives", {})
            keyword_noise = nlp_errors.get("keyword_noise", {})
            topic_quality = nlp_errors.get("topic_quality", {})

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">NER false positives</div>
                        <div class="metric-value">{ner.get("flagged_as_likely_false_positive", "—")}/{ner.get("total_entities", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Keyword noise</div>
                        <div class="metric-value">{keyword_noise.get("flagged_as_discourse_noise", "—")}/{keyword_noise.get("top_keywords_checked", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Avg topic overlap</div>
                        <div class="metric-value">{safe_float(topic_quality.get("average_pairwise_jaccard_similarity")):.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if keyword_noise.get("root_cause"):
                st.caption(keyword_noise["root_cause"])

            flagged_keywords = keyword_noise.get("examples", [])

            if flagged_keywords:

                st.markdown("**Flagged keywords (discourse noise)**")

                for kw in flagged_keywords:
                    st.markdown(
                        f'<span class="tag">{kw.get("term")} ({safe_float(kw.get("score")):.1f})</span>',
                        unsafe_allow_html=True,
                    )

        # ---------------- Action items ----------------

        action_item_errors = error_analysis_data.get("action_items", {})

        with st.expander("Action item errors"):

            unverified = action_item_errors.get("unverified_extractions", {})

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Total extracted</div>
                        <div class="metric-value">{unverified.get("total_extracted_full_meeting", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Verified vs reference</div>
                        <div class="metric-value">{unverified.get("evaluated_against_reference", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Unverified</div>
                        <div class="metric-value">{unverified.get("unverified_count", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if unverified.get("note"):
                st.caption(unverified["note"])

            coverage_gap_examples = action_item_errors.get(
                "missed_items_coverage_gap", {}
            ).get("false_negative_examples", [])

            if coverage_gap_examples:

                st.markdown("**Missed reference items**")

                for ex in coverage_gap_examples:

                    cause = ex.get("likely_cause", "").replace("_", " ").title()

                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-title">{cause}</div>
                            {ex.get("missed_reference_item", "")}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ---------------- Summarization ----------------

        summarization_errors = error_analysis_data.get("summarization", {})

        with st.expander("Summarization errors"):

            length_info = summarization_errors.get("length", {})
            missing_info = summarization_errors.get("missing_info", {})
            unsupported_info = summarization_errors.get(
                "possible_unsupported_details", {}
            )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Compression ratio</div>
                        <div class="metric-value">{length_info.get("compression_ratio", "—")}:1</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Summary words</div>
                        <div class="metric-value">{length_info.get("summary_word_count", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            missing_facts = missing_info.get("missing_facts", [])

            if missing_facts:

                st.markdown(
                    "**Missing facts** (present in the manually-verified "
                    "meeting opening, absent from the generated summary)"
                )

                for fact in missing_facts:
                    st.markdown(
                        f'<span class="tag">{fact.replace("_", " ")}</span>',
                        unsafe_allow_html=True,
                    )

            unsupported_tokens = unsupported_info.get("candidate_tokens", [])

            if unsupported_tokens:

                st.markdown("**Possible unsupported details**")

                for token in unsupported_tokens:
                    st.markdown(
                        f'<span class="tag">{token}</span>',
                        unsafe_allow_html=True,
                    )

            else:

                st.caption(
                    "No unsupported (non-source) proper nouns/numbers "
                    "detected in the summary."
                )
