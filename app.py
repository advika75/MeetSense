import streamlit as st
from pathlib import Path
import base64
import html

st.set_page_config(
    page_title="MeetSense",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
SPEAKER_IMAGE = BASE_DIR / "assets" / "speaker_meeting.png"
ANALYTICS_IMAGE = BASE_DIR / "assets" / "meeting_analytics.png"
WORKSPACE_FILE = BASE_DIR / "app_workspace.py"


def image_b64(path):
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    return ""


speaker = image_b64(SPEAKER_IMAGE)
analytics = image_b64(ANALYTICS_IMAGE)


# ------------------------------------------------------------
# Start button communication
# ------------------------------------------------------------

if "started" not in st.session_state:
    st.session_state.started = False

# Read the Start button redirect BEFORE loading the workspace.
if st.query_params.get("started") == "1":
    st.session_state.started = True

if st.session_state.started and WORKSPACE_FILE.exists():
    workspace = WORKSPACE_FILE.read_text()
    exec(compile(workspace, str(WORKSPACE_FILE), "exec"))
    st.stop()


# ------------------------------------------------------------
# Hide Streamlit interface
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    #MainMenu {visibility:hidden;}
    header {visibility:hidden;}
    footer {visibility:hidden;}

    .block-container {
        padding: 0 !important;
        max-width: none !important;
    }

    [data-testid="stSidebar"] {
        display: none !important;
    }

    [data-testid="stAppViewContainer"] {
        padding: 0 !important;
    }

    iframe {
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Landing page
# ------------------------------------------------------------

speaker_src = (
    f"data:image/png;base64,{speaker}"
    if speaker
    else ""
)

analytics_src = (
    f"data:image/png;base64,{analytics}"
    if analytics
    else ""
)


landing_html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

<style>

.hero-logo {
    width: 58px;
    height: 58px;
    border-radius: 16px;
    background: linear-gradient(135deg, #635bff, #8b5cf6);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 18px;
    box-shadow: 0 10px 25px rgba(98, 88, 238, 0.25);
}

* {
    box-sizing: border-box;
}

html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    font-family: Inter, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
}

body {
    background: #f5f6f8;
    color: #171923;
    transition: background .35s ease, color .35s ease;
}

body.dark {
    background: #111318;
    color: #f5f6f8;
}

.page {
    min-height: 100vh;
    padding: 28px 42px 45px;
}

.nav {
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    max-width: 1450px;
    margin: auto;
}

.brand {
    display: flex;
    align-items: center;
    gap: 11px;
}

.logo {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg,#635bff,#8b5cf6);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    font-weight: 800;
}

.brand-name {
    font-size: 21px;
    font-weight: 800;
    letter-spacing: -.5px;
}

.nav-right {
    display: flex;
    align-items: center;
    gap: 22px;
}

.mode-label {
    font-size: 13px;
    color: #6c7280;
}

.dark .mode-label {
    color: #a8adb8;
}

.toggle {
    width: 48px;
    height: 27px;
    border-radius: 20px;
    background: #d9dce3;
    border: 0;
    padding: 3px;
    cursor: pointer;
    transition: .3s;
}

.toggle-dot {
    width: 21px;
    height: 21px;
    border-radius: 50%;
    background: white;
    transition: .3s;
    box-shadow: 0 2px 6px rgba(0,0,0,.15);
}

.dark .toggle {
    background: #635bff;
}

.dark .toggle-dot {
    transform: translateX(21px);
}

.hero {
    max-width: 1450px;
    min-height: 630px;
    margin: 25px auto 0;

    display: grid;
    grid-template-columns: 52% 48%;

    background: white;
    border: 1px solid #e1e4e9;
    border-radius: 28px;
    overflow: hidden;

    box-shadow: 0 25px 70px rgba(20,25,40,.08);

    transition: background .35s ease, border .35s ease;
}

.dark .hero {
    background: #191c23;
    border-color: #303540;
    box-shadow: 0 25px 70px rgba(0,0,0,.30);
}

.visual {
    position: relative;
    min-height: 630px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: #eef0f3;
    overflow: hidden;

    transition: background .35s ease;
}

.dark .visual {
    background: #20242d;
}

.visual::before {
    content: "";
    position: absolute;

    width: 560px;
    height: 560px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(98,88,238,.13),
            rgba(43,205,235,.05),
            transparent 70%
        );
}

.speaker {
    position: relative;
    width: 78%;
    max-width: 650px;
    z-index: 2;

    mix-blend-mode: multiply;

    transition: transform .4s ease;
}

.speaker:hover {
    transform: scale(1.025);
}

.analytics {
    position: absolute;
    width: 210px;

    right: 28px;
    bottom: 25px;

    z-index: 5;

    border-radius: 18px;

    box-shadow:
        0 20px 45px rgba(0,0,0,.20);

    transition: transform .35s ease;
}

.analytics:hover {
    transform: translateY(-8px);
}

.content {
    padding: 70px 70px;

    display: flex;
    flex-direction: column;
    justify-content: center;
}

.eyebrow {
    width: fit-content;

    padding: 7px 12px;

    border-radius: 100px;

    border: 1px solid #dedff0;
    background: #f4f3ff;

    color: #6258ee;

    font-size: 10px;
    font-weight: 800;

    letter-spacing: 1.2px;
    text-transform: uppercase;

    margin-bottom: 25px;
}

.dark .eyebrow {
    background: #27243e;
    border-color: #403a68;
    color: #958dff;
}

.title {
    font-size: clamp(48px,5vw,68px);
    line-height: 1.02;

    font-weight: 800;
    letter-spacing: -4px;

    color: #171923;
}

.dark .title {
    color: #f5f6f8;
}

.accent {
    color: #6258ee;
}

.dark .accent {
    color: #958dff;
}

.description {
    max-width: 550px;

    margin-top: 25px;

    font-size: 17px;
    line-height: 1.7;

    color: #6b7280;
}

.dark .description {
    color: #a9afba;
}

.features {
    display: grid;
    grid-template-columns: repeat(3,1fr);

    gap: 12px;

    margin-top: 40px;
}

.feature {
    padding: 17px;

    border-radius: 15px;

    border: 1px solid #e2e4e9;

    background: #fafbfc;
}

.dark .feature {
    background: #20242d;
    border-color: #303540;
}

.feature-icon {
    font-size: 18px;
    margin-bottom: 8px;
}

.feature-title {
    font-size: 12px;
    font-weight: 750;

    color: #20232d;
}

.dark .feature-title {
    color: #f1f2f5;
}

.feature-text {
    font-size: 10px;
    line-height: 1.5;

    margin-top: 5px;

    color: #747a86;
}

.dark .feature-text {
    color: #a0a6b1;
}

.bottom {
    max-width: 1450px;

    display: grid;
    grid-template-columns: repeat(3,1fr);

    gap: 18px;

    margin: 18px auto 0;
}

.bottom-card {
    min-height: 120px;

    padding: 22px;

    border-radius: 19px;

    border: 1px solid #e1e4e9;

    background: white;
}

.dark .bottom-card {
    background: #191c23;
    border-color: #303540;
}

.bottom-label {
    font-size: 9px;
    font-weight: 800;

    letter-spacing: 1px;
    text-transform: uppercase;

    color: #6258ee;
}

.bottom-title {
    margin-top: 8px;

    font-size: 16px;
    font-weight: 750;

    color: #20232d;
}

.dark .bottom-title {
    color: #f3f4f6;
}

.bottom-text {
    margin-top: 5px;

    font-size: 11px;

    color: #747a86;
}

.dark .bottom-text {
    color: #a0a6b1;
}

.start {
    display: flex;
    justify-content: center;

    margin-top: -76px;

    position: relative;
    z-index: 20;
}

.start-button {
    width: 155px;
    height: 50px;

    display: flex;
    align-items: center;
    justify-content: center;

    border: 0;
    border-radius: 12px;

    background: #6258ee;
    color: white;

    font-size: 15px;
    font-weight: 750;
    text-decoration: none;

    cursor: pointer;

    box-shadow: 0 12px 28px rgba(98,88,238,.28);

    transition: .25s;
}

.start-button:hover {
    transform: translateY(-2px);
    background: #554be5;
}

@media(max-width:900px) {

    .page {
        padding: 18px;
    }

    .hero {
        grid-template-columns: 1fr;
    }

    .visual {
        min-height: 420px;
    }

    .content {
        padding: 45px 30px;
    }

    .features {
        grid-template-columns: 1fr;
    }

    .bottom {
        grid-template-columns: 1fr;
    }

}

</style>
</head>


<body>

<div class="page">

    <div class="nav">

        <div class="brand">
            <div class="logo">∞</div>
            <div class="brand-name">MeetSense</div>
        </div>

        <div class="nav-right">

            <span class="mode-label" id="modeText">
                Light mode
            </span>

            <button
                class="toggle"
                onclick="toggleMode()"
                aria-label="Toggle dark mode"
            >
                <div class="toggle-dot"></div>
            </button>

        </div>

    </div>

    <div class="hero">

        <div class="visual">

            <img
                class="speaker"
                src="__SPEAKER_SRC__"
            >

            <img
                class="analytics"
                src="__ANALYTICS_SRC__"
            >

        </div>


        <div class="content">

            <div class="eyebrow">
                AI Meeting Intelligence
            </div>

            <div class="title">
                MeetSense
            </div>

            <div class="description">
                Turn meeting recordings into speaker-aware transcripts,
                concise summaries, topics, decisions and actionable tasks.
                One intelligent workspace for everything discussed.
            </div>


            <div class="features">

                <div class="feature">
                    <div class="feature-icon">🎙️</div>
                    <div class="feature-title">
                        Transcribe
                    </div>
                    <div class="feature-text">
                        Accurate speech recognition from meeting audio.
                    </div>
                </div>

                <div class="feature">
                    <div class="feature-icon">👥</div>
                    <div class="feature-title">
                        Identify
                    </div>
                    <div class="feature-text">
                        Know who said what throughout the meeting.
                    </div>
                </div>

                <div class="feature">
                    <div class="feature-icon">✦</div>
                    <div class="feature-title">
                        Analyze
                    </div>
                    <div class="feature-text">
                        Surface summaries, topics and action items.
                    </div>
                </div>

            </div>

        </div>

    </div>


    <div class="bottom">

    </div>


    <div class="start">
        <a class="start-button" id="start-btn" href="#">
            Start&nbsp;&nbsp;→
        </a>
    </div>

</div>


<script>

function toggleMode() {

    document.body.classList.toggle("dark");

    const dark =
        document.body.classList.contains("dark");

    document.getElementById("modeText").innerText =
        dark ? "Dark mode" : "Light mode";

}


document.getElementById("start-btn").addEventListener("click", function(e) {
    e.preventDefault();
    // components.html()'s iframe is sandboxed without allow-top-navigation,
    // so it can never navigate window.parent directly. Instead, reach into
    // the parent document (allowed because allow-same-origin is set) and
    // click the real Streamlit button that lives outside the sandbox.
    const buttons = window.parent.document.querySelectorAll("button");
    for (const btn of buttons) {
        if (btn.textContent.trim() === "ENTER_MEETSENSE") {
            btn.click();
            break;
        }
    }
});

</script>

</body>
</html>
"""

# Insert the two local landing-page images safely.
landing_html = landing_html.replace("__SPEAKER_SRC__", speaker_src)
landing_html = landing_html.replace("__ANALYTICS_SRC__", analytics_src)

# ------------------------------------------------------------
# Render the actual HTML inside an iframe.
# ------------------------------------------------------------

import streamlit.components.v1 as components

components.html(
    landing_html,
    height=930,
    scrolling=True,
)


# ------------------------------------------------------------
# Hidden Streamlit button used to receive the start action.
# ------------------------------------------------------------

if st.button(
    "ENTER_MEETSENSE",
    key="hidden_start",
):
    st.session_state.started = True
    st.rerun()


st.markdown(
    """
    <style>
    button[kind="secondary"] {
        position: fixed;
        left: -1000px;
        top: -1000px;
        opacity: 0;
        pointer-events: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
