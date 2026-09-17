import streamlit as st
from pathlib import Path
import base64

# ============================================================
# MEETSENSE — LANDING PAGE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SPEAKER_IMAGE = BASE_DIR / "assets" / "speaker_meeting.png"
ANALYTICS_IMAGE = BASE_DIR / "assets" / "meeting_analytics.png"
WORKSPACE_FILE = BASE_DIR / "app_workspace.py"

st.set_page_config(
    page_title="MeetSense",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "started" not in st.session_state:
    st.session_state.started = False

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# ============================================================
# If Start has been clicked, launch the existing workspace
# ============================================================

if st.session_state.started:

    workspace_code = WORKSPACE_FILE.read_text()

    # Execute the original MeetSense application.
    exec(compile(workspace_code, str(WORKSPACE_FILE), "exec"))

    st.stop()


# ============================================================
# LANDING PAGE
# ============================================================

dark = st.session_state.dark_mode

if dark:
    BG = "#111318"
    PANEL = "#191c23"
    SOFT = "#20242d"
    TEXT = "#f5f6f8"
    MUTED = "#a4a9b5"
    BORDER = "#303540"
    ACCENT = "#6d63ff"
else:
    BG = "#f6f7f9"
    PANEL = "#ffffff"
    SOFT = "#eef0f4"
    TEXT = "#171a22"
    MUTED = "#666d79"
    BORDER = "#e1e4e9"
    ACCENT = "#6258ee"


def image_to_base64(path):
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()


speaker_b64 = image_to_base64(SPEAKER_IMAGE)
analytics_b64 = image_to_base64(ANALYTICS_IMAGE)


# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* Remove Streamlit chrome */

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    header {{
        background: transparent !important;
    }}

    [data-testid="stSidebar"] {{
        display: none;
    }}

    .stApp {{
        background: {BG};
    }}

    .block-container {{
        max-width: 1500px !important;
        padding: 0 !important;
    }}

    /* -----------------------------------------
       Overall page
       ----------------------------------------- */

    .ms-page {{
        min-height: 100vh;
        padding: 28px 44px 50px 44px;
        box-sizing: border-box;
        color: {TEXT};
    }}

    /* -----------------------------------------
       Navigation
       ----------------------------------------- */

    .ms-nav {{
        height: 62px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 32px;
    }}

    .ms-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    .ms-logo {{
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: linear-gradient(135deg, #635bff, #8b5cf6);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 21px;
        font-weight: 800;
    }}

    .ms-brand-name {{
        font-size: 21px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }}

    .ms-nav-label {{
        font-size: 13px;
        color: {MUTED};
    }}

    /* -----------------------------------------
       Main hero
       ----------------------------------------- */

    .ms-hero {{
        max-width: 1420px;
        min-height: 625px;
        margin: 0 auto;
        display: grid;
        grid-template-columns: 52% 48%;
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 28px;
        overflow: hidden;
        box-shadow: 0 25px 70px rgba(20,25,40,.08);
    }}

    /* -----------------------------------------
       Left visual
       ----------------------------------------- */

    .ms-visual {{
        position: relative;
        min-height: 625px;
        background: {SOFT};
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }}

    .ms-visual::before {{
        content: "";
        position: absolute;
        width: 500px;
        height: 500px;
        border-radius: 50%;
        background:
            radial-gradient(
                circle,
                rgba(98,88,238,.13),
                rgba(45,210,235,.05),
                transparent 70%
            );
    }}

    .ms-speaker {{
        position: relative;
        width: 78%;
        max-width: 650px;
        z-index: 2;
        mix-blend-mode: multiply;
    }}

    .ms-analytics {{
        position: absolute;
        width: 210px;
        right: 30px;
        bottom: 25px;
        z-index: 4;
        border-radius: 18px;
        box-shadow: 0 16px 45px rgba(15,23,42,.20);
    }}

    /* -----------------------------------------
       Right content
       ----------------------------------------- */

    .ms-content {{
        padding: 75px 70px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
    }}

    .ms-eyebrow {{
        width: fit-content;
        padding: 7px 12px;
        border-radius: 999px;
        border: 1px solid {BORDER};
        background: {SOFT};
        color: {ACCENT};
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 24px;
    }}

    .ms-title {{
        font-size: clamp(52px, 5vw, 78px);
        line-height: 1.02;
        letter-spacing: -4px;
        font-weight: 800;
        color: {TEXT};
        margin: 0;
    }}

    .ms-title-accent {{
        color: {ACCENT};
    }}

    .ms-description {{
        max-width: 560px;
        margin-top: 24px;
        font-size: 17px;
        line-height: 1.7;
        color: {MUTED};
    }}

    /* -----------------------------------------
       Feature cards
       ----------------------------------------- */

    .ms-features {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 42px;
    }}

    .ms-feature {{
        border: 1px solid {BORDER};
        background: {PANEL};
        border-radius: 15px;
        padding: 17px;
    }}

    .ms-feature-icon {{
        font-size: 18px;
        margin-bottom: 8px;
    }}

    .ms-feature-title {{
        color: {TEXT};
        font-size: 12px;
        font-weight: 750;
    }}

    .ms-feature-text {{
        color: {MUTED};
        font-size: 10px;
        line-height: 1.5;
        margin-top: 5px;
    }}

    /* -----------------------------------------
       Bottom information cards
       ----------------------------------------- */

    .ms-bottom {{
        max-width: 1420px;
        margin: 18px auto 0 auto;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
    }}

    .ms-bottom-card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 19px;
        padding: 22px;
        min-height: 115px;
    }}

    .ms-bottom-label {{
        color: {ACCENT};
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    .ms-bottom-title {{
        color: {TEXT};
        font-size: 16px;
        font-weight: 750;
        margin-top: 8px;
    }}

    .ms-bottom-text {{
        color: {MUTED};
        font-size: 11px;
        margin-top: 5px;
    }}

    @media(max-width: 900px) {{

        .ms-page {{
            padding: 18px;
        }}

        .ms-hero {{
            grid-template-columns: 1fr;
        }}

        .ms-visual {{
            min-height: 420px;
        }}

        .ms-content {{
            padding: 45px 30px;
        }}

        .ms-features {{
            grid-template-columns: 1fr;
        }}

        .ms-bottom {{
            grid-template-columns: 1fr;
        }}

    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE HTML
# ============================================================

analytics_html = ""

if analytics_b64:
    analytics_html = f"""
        <img
            class="ms-analytics"
            src="data:image/png;base64,{analytics_b64}"
        >
    """

speaker_html = ""

if speaker_b64:
    speaker_html = f"""
        <img
            class="ms-speaker"
            src="data:image/png;base64,{speaker_b64}"
        >
    """


st.markdown(
    f"""
    <div class="ms-page">

        <div class="ms-nav">

            <div class="ms-brand">
                <div class="ms-logo">∞</div>
                <div class="ms-brand-name">MeetSense</div>
            </div>

            <div class="ms-nav-label">
                Speaker-aware meeting intelligence
            </div>

        </div>


        <div class="ms-hero">

            <div class="ms-visual">

                {speaker_html}

                {analytics_html}

            </div>


            <div class="ms-content">

                <div class="ms-eyebrow">
                    AI Meeting Intelligence
                </div>

                <div class="ms-title">
                    Meetings,<br>
                    <span class="ms-title-accent">understood.</span>
                </div>

                <div class="ms-description">
                    Turn meeting recordings into speaker-aware transcripts,
                    concise summaries, topics, decisions and actionable tasks.
                    One intelligent workspace for everything discussed.
                </div>

                <div class="ms-features">

                    <div class="ms-feature">
                        <div class="ms-feature-icon">🎙️</div>
                        <div class="ms-feature-title">
                            Transcribe
                        </div>
                        <div class="ms-feature-text">
                            Accurate speech recognition.
                        </div>
                    </div>

                    <div class="ms-feature">
                        <div class="ms-feature-icon">👥</div>
                        <div class="ms-feature-title">
                            Identify
                        </div>
                        <div class="ms-feature-text">
                            Know who said what.
                        </div>
                    </div>

                    <div class="ms-feature">
                        <div class="ms-feature-icon">✦</div>
                        <div class="ms-feature-title">
                            Analyze
                        </div>
                        <div class="ms-feature-text">
                            Find what matters.
                        </div>
                    </div>

                </div>

            </div>

        </div>


        <div class="ms-bottom">

            <div class="ms-bottom-card">
                <div class="ms-bottom-label">
                    01 · Transcription
                </div>
                <div class="ms-bottom-title">
                    Know what was said.
                </div>
                <div class="ms-bottom-text">
                    Timestamped transcripts generated directly from meeting audio.
                </div>
            </div>

            <div class="ms-bottom-card">
                <div class="ms-bottom-label">
                    02 · Speaker Intelligence
                </div>
                <div class="ms-bottom-title">
                    Know who said it.
                </div>
                <div class="ms-bottom-text">
                    Connect spoken content with individual speakers.
                </div>
            </div>

            <div class="ms-bottom-card">
                <div class="ms-bottom-label">
                    03 · Meeting Intelligence
                </div>
                <div class="ms-bottom-title">
                    Know what matters.
                </div>
                <div class="ms-bottom-text">
                    Surface summaries, decisions, topics and next steps.
                </div>
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# START BUTTON
# ============================================================

_, start_col, _ = st.columns([2.8, 1.2, 2.8])

with start_col:
    if st.button(
        "Start  →",
        key="start_meetsense",
        use_container_width=True,
    ):
        st.session_state.started = True
        st.rerun()


st.markdown(
    f"""
    <style>

    div[data-testid="stButton"] > button {{
        height: 52px;
        border-radius: 13px;
        border: 1px solid {ACCENT};
        background: {ACCENT};
        color: white;
        font-size: 15px;
        font-weight: 700;
        margin-top: -2px;
    }}

    div[data-testid="stButton"] > button:hover {{
        border-color: {ACCENT};
        color: white;
        filter: brightness(.92);
    }}

    </style>
    """,
    unsafe_allow_html=True,
)
