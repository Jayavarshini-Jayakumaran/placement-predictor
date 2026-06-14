"""
app.py — Campus Placement Eligibility Predictor
"""

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from features import add_engineered_features, FEATURE_COLUMNS, cgpa_band_label

# ── Colors ──────────────────────────────────────────────────────────
DARK_BEIGE      = "#D9CDB4"
PANEL_BEIGE     = "#C8BA9E"
ALGAE_GREEN     = "#5C7A4E"
ALGAE_DARK      = "#43603A"
LIGHT_GREEN     = "#A9D08E"
SOFT_RED        = "#C97B63"
TEXT_DARK       = "#2E2B22"
LIGHT_BROWN     = "#A68B5B"

# ── SVG Icons ────────────────────────────────────────────────────────
ICONS = {
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "check":  '<circle cx="12" cy="12" r="10"/><polyline points="8 12 11 15 16 9"/>',
    "cross":  '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
    "idea":   '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"/>',
    "user":   '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
}

def svg_icon(name, color=None, size=22):
    c = color or ALGAE_DARK
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:middle;margin-right:7px">{ICONS.get(name,"")}</svg>'
    )

def section_heading(icon_name, text, level="h2", color=None):
    c = color or ALGAE_DARK
    st.markdown(
        f'<{level} style="color:{c};display:flex;align-items:center;'
        f'margin-bottom:0.25em">{svg_icon(icon_name,c)}{text}</{level}>',
        unsafe_allow_html=True,
    )

# ── Page config & CSS ────────────────────────────────────────────────
st.set_page_config(page_title="Placement Predictor", layout="centered")

st.markdown(f"""
<style>
/* Hide the +/- spinner buttons on number inputs */
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {{ -webkit-appearance:none; margin:0; }}
input[type=number] {{ -moz-appearance:textfield; }}

/* App background */
.stApp, section[data-testid="stSidebar"], .block-container {{
    background-color: {DARK_BEIGE};
    color: {TEXT_DARK};
}}

/* Input field border — light brown */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {{
    border: 1.5px solid {LIGHT_BROWN} !important;
    border-radius: 7px !important;
    background-color: {PANEL_BEIGE} !important;
    color: {TEXT_DARK} !important;
}}

/* Hide the stepper container entirely */
div[data-testid="stNumberInput"] > div > div:last-child {{ display:none !important; }}

/* Predict button */
.stButton > button {{
    background-color: {ALGAE_GREEN};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55em 1.6em;
    font-weight: 600;
    font-size: 1em;
}}
.stButton > button:hover {{ background-color: {ALGAE_DARK}; color: white; }}

/* Result boxes */
.res-eligible {{
    background-color:{LIGHT_GREEN}; color:{ALGAE_DARK};
    border-radius:10px; padding:0.9em 1.2em;
    font-size:1.05em; font-weight:600; margin:0.5em 0;
}}
.res-not {{
    background-color:#EDD5C8; color:{SOFT_RED};
    border-radius:10px; padding:0.9em 1.2em;
    font-size:1.05em; font-weight:600; margin:0.5em 0;
}}

/* Insight cards */
.card {{
    background-color:{PANEL_BEIGE};
    border-left:4px solid {ALGAE_GREEN};
    border-radius:7px;
    padding:0.65em 1em;
    margin:0.35em 0;
    color:{TEXT_DARK};
}}
.card-warn {{
    background-color:{PANEL_BEIGE};
    border-left:4px solid {SOFT_RED};
    border-radius:7px;
    padding:0.65em 1em;
    margin:0.35em 0;
    color:{TEXT_DARK};
}}
.slogan {{
    background-color:{ALGAE_DARK};
    color:white;
    border-radius:9px;
    padding:0.8em 1.2em;
    margin-top:0.8em;
    font-size:1em;
    font-style:italic;
    font-weight:500;
}}
label, .stMarkdown p {{ color:{TEXT_DARK} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ──────────────────────────────────────────────
model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ── Header ───────────────────────────────────────────────────────────
section_heading("target", "Campus Placement Eligibility Predictor", level="h1")
st.write(
    "Enter your academic details below and select **Predict** to instantly "
    "see your placement eligibility with a confidence score and "
    "personalised feedback."
)

st.markdown("---")

# ── Inputs ───────────────────────────────────────────────────────────
section_heading("user", "Your Details")

col1, col2 = st.columns(2)

with col1:
    cgpa_str = st.text_input(
        "CGPA",
        value="7.00",
        placeholder="0.0 – 10.0",
        help="Your current cumulative GPA on a 10-point scale.",
    )
    cgpa, cgpa_ok = None, True
    if cgpa_str.strip():
        try:
            cgpa = float(cgpa_str)
            if not (0.0 <= cgpa <= 10.0):
                st.error("CGPA must be between 0.0 and 10.0.")
                cgpa_ok = False
        except ValueError:
            st.error("CGPA must be a number (e.g. 7.85).")
            cgpa_ok = False

    skills_str = st.text_input(
        "Skill Certifications",
        value="2",
        placeholder="1 – 15",
        help="How many skill certification courses have you completed?",
    )
    skills, skills_ok = None, True
    if skills_str.strip():
        try:
            skills = int(skills_str)
            if not (1 <= skills <= 15):
                st.error("Skill Certifications must be a whole number between 1 and 15.")
                skills_ok = False
        except ValueError:
            st.error("Skill Certifications must be a whole number (e.g. 3).")
            skills_ok = False

with col2:
    internships_str = st.text_input(
        "Internships Completed",
        value="1",
        placeholder="0 – 15",
        help="How many internships have you completed?",
    )
    internships, internships_ok = None, True
    if internships_str.strip():
        try:
            internships = int(internships_str)
            if not (0 <= internships <= 15):
                st.error("Internships must be a whole number between 0 and 15.")
                internships_ok = False
        except ValueError:
            st.error("Internships must be a whole number (e.g. 1).")
            internships_ok = False

    projects_str = st.text_input(
        "Projects Completed",
        value="2",
        placeholder="Any positive number",
        help="How many academic or personal projects have you completed?",
    )
    projects, projects_ok = None, True
    if projects_str.strip():
        try:
            projects = int(projects_str)
            if projects < 0:
                st.error("Projects must be 0 or more.")
                projects_ok = False
        except ValueError:
            st.error("Projects must be a whole number (e.g. 4).")
            projects_ok = False

predict_clicked = st.button("Predict")

# ── Prediction (only runs if all fields valid) ────────────────────────
all_valid = all([cgpa_ok, skills_ok, internships_ok, projects_ok,
                 cgpa is not None, skills is not None,
                 internships is not None, projects is not None])

if predict_clicked:
    if not all_valid:
        st.warning("Please fix the errors above before predicting.")
    else:
        input_df = pd.DataFrame([{
            "cgpa": cgpa, "skills": skills,
            "internships": internships, "projects": projects
        }])
        input_df    = add_engineered_features(input_df)
        X_scaled    = scaler.transform(input_df[FEATURE_COLUMNS])
        pred        = model.predict(X_scaled)[0]
        proba       = model.predict_proba(X_scaled)[0]
        confidence  = proba[pred] * 100
        eligible_pct = proba[1] * 100

        st.markdown("---")
        section_heading("check" if pred == 1 else "cross", "Your Result")

        if pred == 1:
            st.markdown(
                f'<div class="res-eligible">'
                f'{svg_icon("check", ALGAE_DARK)}'
                f'You are <b>eligible</b> for campus placement based on your profile. '
                f'(Confidence: {confidence:.0f}%)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="res-not">'
                f'{svg_icon("cross", SOFT_RED)}'
                f'Based on your current profile, you are <b>not yet eligible</b> '
                f'for campus placement. (Confidence: {confidence:.0f}%)</div>',
                unsafe_allow_html=True,
            )

        # ── Confidence bar ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 1.0))
        fig.patch.set_facecolor(DARK_BEIGE)
        ax.set_facecolor(DARK_BEIGE)
        ax.barh([0], [100], color=PANEL_BEIGE)
        ax.barh([0], [eligible_pct], color=LIGHT_GREEN if eligible_pct >= 50 else SOFT_RED)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Likelihood of being eligible (%)", color=TEXT_DARK)
        ax.tick_params(colors=TEXT_DARK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(min(eligible_pct + 1, 95), 0, f"{eligible_pct:.0f}%",
                va="center", fontweight="bold", color=TEXT_DARK)
        st.pyplot(fig)

        # ── Personalised insights ──────────────────────────────────
        st.markdown("---")
        section_heading("idea", "Suggestions for You")

        band  = cgpa_band_label(input_df["cgpa_band"].iloc[0])
        tips  = []

        if cgpa < 6.5:
            tips.append(("warn", f"Your CGPA of {cgpa:.2f} is in the <b>low</b> range. "
                         "Bringing it above 7.0 would have the biggest positive impact on your eligibility."))
        elif cgpa < 8.5:
            tips.append(("ok", f"Your CGPA of {cgpa:.2f} is in the <b>medium</b> range. "
                         "Pushing towards 8.5+ can further strengthen your chances."))
        else:
            tips.append(("ok", f"Your CGPA of {cgpa:.2f} is in the <b>high</b> range. "
                         "This is one of your strongest assets."))

        if skills < 4:
            tips.append(("warn", f"You have completed <b>{skills}</b> skill certification(s). "
                         "Earning a few more certifications in relevant technologies will strengthen your profile."))
        else:
            tips.append(("ok", f"You have <b>{skills}</b> skill certification(s). "
                         "This shows great initiative — keep adding relevant ones."))

        if internships < 1:
            tips.append(("warn", "You have <b>no internships</b> yet. "
                         "Completing even one internship makes a significant difference to recruiters."))
        else:
            tips.append(("ok", f"You have completed <b>{internships}</b> internship(s). "
                         "Real-world experience is a strong signal to employers."))

        if projects < 2:
            tips.append(("warn", f"You have completed <b>{projects}</b> project(s). "
                         "Working on at least 2 projects will help demonstrate your practical skills to recruiters."))
        else:
            tips.append(("ok", f"You have completed <b>{projects}</b> project(s). "
                         "A solid project portfolio shows you can apply what you learn — the more, the stronger."))

        for kind, msg in tips:
            css_class = "card-warn" if kind == "warn" else "card"
            st.markdown(f'<div class="{css_class}">{msg}</div>', unsafe_allow_html=True)

        # ── Closing slogan ─────────────────────────────────────────
        has_gaps = any(k == "warn" for k, _ in tips)
        if has_gaps:
            slogan = "Consistency doesn't ask for perfection — it just asks you to show up every single day."
        else:
            slogan = "You built this, one day at a time. Keep that standard. Consistency is the only crown that never slips."

        st.markdown(f'<div class="slogan">{slogan}</div>', unsafe_allow_html=True)
