from pathlib import Path
import base64
import streamlit as st
import streamlit.components.v1 as components

landing_html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

<style>

* {
    box-sizing: border-box;
}

html,
body {
    margin: 0;
    padding: 0;
    width: 100%;
    min-height: 100%;
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

body {
    background: #ffffff;
    color: #111318;
}

.page {
    width: 100%;
    min-height: 900px;
    padding: 28px 38px 42px;
}

/* ----------------------------------------------------------
   TOP BRAND
---------------------------------------------------------- */

.topbar {
    height: 72px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    max-width: 1450px;
    margin: 0 auto 28px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-logo {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #635bff, #8b5cf6);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 23px;
    font-weight: 800;
}

.brand-name {
    font-size: 21px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.mode {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #6f7480;
    font-size: 13px;
}

/* ----------------------------------------------------------
   MAIN LANDING CARD
---------------------------------------------------------- */

.landing {
    max-width: 1450px;
    min-height: 730px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 1fr 1fr;
    background: #ffffff;
    border: 1px solid #e1e4ea;
    border-radius: 28px;
    overflow: hidden;
    box-shadow: 0 18px 55px rgba(20, 25, 45, 0.08);
}

/* ----------------------------------------------------------
   LEFT IMAGE
---------------------------------------------------------- */

.left {
    background: #d4d4d6;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 730px;
    position: relative;
    overflow: hidden;
}

.speaker-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

/* Analytics image floating over the speaker image */

.analytics-card {
    position: absolute;
    right: 42px;
    bottom: 48px;
    width: 235px;
    height: 175px;
    background: #ffffff;
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 18px 38px rgba(0, 0, 0, 0.18);
    display: flex;
    align-items: center;
    justify-content: center;
}

.analytics-image {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* ----------------------------------------------------------
   RIGHT CONTENT
---------------------------------------------------------- */

.right {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 70px 82px;
    background: #ffffff;
}

.hero-logo {
    width: 70px;
    height: 70px;
    border-radius: 18px;
    background: linear-gradient(135deg, #635bff, #8b5cf6);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 26px;
    box-shadow: 0 14px 30px rgba(99, 91, 255, 0.25);
}

.title {
    font-size: clamp(54px, 5vw, 82px);
    line-height: 0.95;
    letter-spacing: -4px;
    font-weight: 800;
    margin: 0 0 22px;
    color: #111318;
}

.subtitle {
    font-size: 24px;
    line-height: 1.45;
    color: #20232b;
    margin-bottom: 0;
}

/* ----------------------------------------------------------
   START BUTTON
---------------------------------------------------------- */

.start-area {
    margin-top: 70px;
}

.start-button {
    width: 210px;
    height: 62px;
    border: 1px solid #15171c;
    border-radius: 4px;
    background: #ffffff;
    color: #111318;
    font-size: 17px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
}

.start-button:hover {
    background: #111318;
    color: #ffffff;
    transform: translateY(-2px);
}

/* ----------------------------------------------------------
   RESPONSIVE
---------------------------------------------------------- */

@media (max-width: 900px) {

    .page {
        padding: 18px;
    }

    .landing {
        grid-template-columns: 1fr;
    }

    .left {
        min-height: 480px;
    }

    .right {
        min-height: 500px;
        padding: 50px 38px;
    }

    .analytics-card {
        width: 180px;
        height: 135px;
        right: 24px;
        bottom: 24px;
    }

    .title {
        font-size: 58px;
    }
}

</style>
</head>

<body>

<div class="page">

    <div class="topbar">

        <div class="brand">

            <div class="brand-logo">
                ∞
            </div>

            <div class="brand-name">
                MeetSense
            </div>

        </div>

        <div class="mode">
            Light mode
        </div>

    </div>


    <div class="landing">

        <!-- LEFT SIDE -->

        <div class="left">

            <img
                class="speaker-image"
                src="__SPEAKER_IMAGE__"
            >

            <div class="analytics-card">

                <img
                    class="analytics-image"
                    src="__ANALYTICS_IMAGE__"
                >

            </div>

        </div>


        <!-- RIGHT SIDE -->

        <div class="right">

            <div class="hero-logo">
                ∞
            </div>

            <h1 class="title">
                MeetSense
            </h1>

            <div class="subtitle">
                Meeting intelligence, simplified
            </div>

            <div class="start-area">

                <button
                    class="start-button"
                    onclick="startMeetSense()"
                >
                    Start
                </button>

            </div>

        </div>

    </div>

</div>


<script>

function startMeetSense() {

    /*
       Tell the parent Streamlit page that the user
       wants to enter the application.
    */

    window.parent.postMessage(
        {
            type: "meetsense_start"
        },
        "*"
    );

}

</script>

</body>
</html>
"""

landing_html = landing_html.replace(
    "__SPEAKER_IMAGE__",
    speaker_src
)

landing_html = landing_html.replace(
    "__ANALYTICS_IMAGE__",
    analytics_src
)

    
# Insert the two local images safely after the HTML string is created.
landing_html = landing_html.replace("{speaker_src}", speaker_src)
landing_html = landing_html.replace("{analytics_src}", analytics_src)






