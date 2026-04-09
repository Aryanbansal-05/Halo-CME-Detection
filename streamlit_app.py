import streamlit as st
import pandas as pd
import numpy as np
import joblib
from app.Utils.features import extract_features_from_window
import plotly.graph_objects as go
import plotly.express as px

# ==========================
# Load Model
# ==========================
MODEL_PATH = "app/model/cme_model.joblib"
model = joblib.load(MODEL_PATH)
THRESHOLD = 0.45

# ==========================
# Page Config (must be first)
# ==========================
st.set_page_config(
    page_title="Halo CME Detection",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Dark Mode CSS
# ==========================
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* ── Global Reset & Base ── */
    *, *::before, *::after { box-sizing: border-box; }

    html, body, .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        background-color: #000000 !important;
        color: #f5f5f7;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: rgba(28,28,30,0.95) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.5rem;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.875rem !important;
        font-weight: 400;
        color: #f5f5f7 !important;
        padding: 0.5rem 0;
    }
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        color: #98989d !important;
        font-size: 0.8rem;
    }

    /* ── Main Content Area ── */
    .main .block-container {
        padding: 2.5rem 3rem 4rem 3rem !important;
        max-width: 1200px;
        background-color: #000000 !important;
    }

    /* ── Typography ── */
    h1 { font-size: 2.75rem !important; font-weight: 600 !important; letter-spacing: -0.025em !important; color: #f5f5f7 !important; line-height: 1.1 !important; text-shadow: none !important; }
    h2 { font-size: 1.75rem !important; font-weight: 600 !important; letter-spacing: -0.02em !important; color: #f5f5f7 !important; text-shadow: none !important; }
    h3 { font-size: 1.25rem !important; font-weight: 500 !important; letter-spacing: -0.01em !important; color: #f5f5f7 !important; text-shadow: none !important; }
    p  { color: #98989d !important; line-height: 1.6 !important; font-size: 0.9375rem !important; }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid rgba(255,255,255,0.08) !important; margin: 2rem 0 !important; }

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {
        background: #1c1c1e;
        border-radius: 18px;
        padding: 1.5rem !important;
        border: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #f5f5f7 !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8125rem !important;
        font-weight: 500 !important;
        color: #98989d !important;
        text-transform: uppercase;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.8125rem !important;
    }

    /* ── Alerts / Info boxes ── */
    .stAlert {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-left: 1px solid rgba(255,255,255,0.06) !important;
        background: #1c1c1e !important;
        color: #f5f5f7 !important;
        padding: 1rem 1.25rem !important;
        font-size: 0.875rem !important;
        box-shadow: none !important;
    }
    .stAlert p { color: #f5f5f7 !important; font-size: 0.875rem !important; }

    /* ── Success / Error / Info states ── */
    div[data-testid="stSuccess"] {
        background: rgba(52,199,89,0.1) !important;
        border-color: rgba(52,199,89,0.25) !important;
    }
    div[data-testid="stError"] {
        background: rgba(255,59,48,0.1) !important;
        border-color: rgba(255,59,48,0.25) !important;
    }
    div[data-testid="stInfo"] {
        background: rgba(10,132,255,0.1) !important;
        border-color: rgba(10,132,255,0.2) !important;
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        border: 1.5px dashed rgba(255,255,255,0.15) !important;
        border-radius: 18px !important;
        background: #1c1c1e !important;
        padding: 1.5rem !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(10,132,255,0.5) !important;
    }
    [data-testid="stFileUploader"] label {
        font-size: 0.9375rem !important;
        color: #f5f5f7 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        font-size: 0.875rem !important;
        color: #98989d !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #0a84ff !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 980px !important;
        padding: 0.6rem 1.5rem !important;
        font-size: 0.9375rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
        transition: background 0.2s ease, transform 0.15s ease !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        background: #1a8fff !important;
        transform: none !important;
        box-shadow: none !important;
    }
    .stButton > button:active {
        background: #0070e0 !important;
        transform: scale(0.98) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #2c2c2e !important;
        border-radius: 10px !important;
        padding: 3px !important;
        gap: 2px !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: #98989d !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: #3a3a3c !important;
        color: #f5f5f7 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        overflow: hidden !important;
    }
    iframe[title="st.dataframe"] {
        border-radius: 14px !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        background: #1c1c1e !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.9375rem !important;
        font-weight: 500 !important;
        color: #f5f5f7 !important;
        padding: 0.875rem 1.25rem !important;
    }

    /* ── Radio buttons in sidebar ── */
    .stRadio > label {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #f5f5f7 !important;
    }
    .stRadio [role="radiogroup"] {
        gap: 4px !important;
    }
    .stRadio [data-testid="stMarkdownContainer"] > p {
        font-size: 0.875rem !important;
        padding: 6px 12px !important;
        border-radius: 8px !important;
        color: #f5f5f7 !important;
        cursor: pointer;
        transition: background 0.15s ease;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
</style>
""", unsafe_allow_html=True)


# ─── Helper: Dark card wrapper ───────────────────────────────────────
def apple_card(content_html, padding="1.5rem 1.75rem"):
    return f"""
    <div style="
        background: #1c1c1e;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.06);
        padding: {padding};
        margin-bottom: 1rem;
    ">{content_html}</div>
    """

def badge(text, color="#0a84ff", bg="rgba(10,132,255,0.12)"):
    return f"""<span style="
        display:inline-block;
        background:{bg};
        color:{color};
        font-size:0.75rem;
        font-weight:500;
        padding:3px 10px;
        border-radius:980px;
        letter-spacing:0.02em;
    ">{text}</span>"""


# ==========================
# Sidebar
# ==========================
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <div style="
            width: 44px; height: 44px;
            background: linear-gradient(135deg, #ff9500, #ff6b00);
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.4rem;
            margin-bottom: 1rem;
        ">☀️</div>
        <p style="font-size:1.1rem; font-weight:600; color:#f5f5f7 !important; margin:0; line-height:1.2;">CME Detection</p>
        <p style="font-size:0.8rem; color:#98989d !important; margin:4px 0 0 0;">Aditya-L1 · SWIS L2</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ("Prediction Interface", "How It Works", "About the Model"),
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top:2rem; height:1px; background:rgba(255,255,255,0.06);'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 1.5rem;">
        <p style="font-size:0.75rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.75rem;">Developer</p>
        <p style="font-size:0.9375rem; font-weight:500; color:#f5f5f7 !important; margin:0;">Aryan Bansal</p>
        <p style="font-size:0.8125rem; color:#98989d !important; margin:2px 0 0 0;">3rd Year · Thapar University</p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# ☀️  PREDICTION PAGE
# =====================================================
if page == "Prediction Interface":

    # ── Hero ──────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom: 2.5rem;">
        <p style="font-size:0.8125rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Space Weather · AI Detection</p>
        <h1 style="margin:0 0 0.75rem 0; color:#f5f5f7 !important;">Halo CME<br>Detection System</h1>
        <p style="font-size:1.0625rem; color:#98989d !important; max-width:520px; margin:0; line-height:1.55;">
            Physics-informed machine learning for real-time solar wind plasma analysis.
            Upload your SWIS data to begin.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Spec pills ────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(apple_card("""
        <p style="font-size:0.75rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.06em; margin:0 0 0.25rem 0;">Data Source</p>
        <p style="font-size:1rem; font-weight:500; color:#f5f5f7 !important; margin:0;">Aditya-L1 SWIS L2</p>
        """, padding="1.25rem 1.5rem"), unsafe_allow_html=True)
    with col2:
        st.markdown(apple_card("""
        <p style="font-size:0.75rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.06em; margin:0 0 0.25rem 0;">Resolution</p>
        <p style="font-size:1rem; font-weight:500; color:#f5f5f7 !important; margin:0;">5-minute intervals</p>
        """, padding="1.25rem 1.5rem"), unsafe_allow_html=True)
    with col3:
        st.markdown(apple_card("""
        <p style="font-size:0.75rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.06em; margin:0 0 0.25rem 0;">Method</p>
        <p style="font-size:1rem; font-weight:500; color:#f5f5f7 !important; margin:0;">Physics-informed ML</p>
        """, padding="1.25rem 1.5rem"), unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────
    st.markdown("""
    <p style="font-size:0.8125rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.75rem;">Upload Data</p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Select a CSV file containing solar wind plasma parameters",
        type=["csv"],
        help="Required columns: proton_density, proton_speed, proton_temperature, alpha_density"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            tab1, tab2, tab3 = st.tabs(["  Data Preview  ", "  Detection Result  ", "  Feature Analysis  "])

            with tab1:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Records", f"{len(df):,}")
                with col_b:
                    st.metric("Columns", len(df.columns))
                st.dataframe(df.head(20), use_container_width=True, height=380)

            required_cols = {"proton_density", "proton_speed", "proton_temperature", "alpha_density"}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                features_df = extract_features_from_window(df)

                if features_df.isnull().values.any():
                    st.error("Insufficient data for feature computation. At least ~15 minutes of data is required.")
                else:
                    prob = model.predict_proba(features_df)[0][1]
                    prediction = "CME Detected" if prob >= THRESHOLD else "No CME"
                    is_cme = prob >= THRESHOLD

                    with tab2:
                        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

                        # ── Gauge ──────────────────────────────────
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=round(prob * 100, 1),
                            number={
                                "suffix": "%",
                                "font": {"size": 48, "color": "#f5f5f7", "family": "-apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif"},
                            },
                            title={
                                "text": "Confidence Score",
                                "font": {"size": 13, "color": "#98989d", "family": "-apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif"}
                            },
                            gauge={
                                "axis": {
                                    "range": [0, 100],
                                    "tickwidth": 0,
                                    "tickcolor": "rgba(0,0,0,0)",
                                    "tickfont": {"size": 11, "color": "#98989d"},
                                    "nticks": 6,
                                },
                                "bar": {"color": "#ff453a" if is_cme else "#30d158", "thickness": 0.28},
                                "bgcolor": "rgba(0,0,0,0)",
                                "borderwidth": 0,
                                "steps": [
                                    {"range": [0, THRESHOLD * 100], "color": "rgba(48,209,88,0.08)"},
                                    {"range": [THRESHOLD * 100, 100], "color": "rgba(255,69,58,0.08)"},
                                ],
                                "threshold": {
                                    "line": {"color": "#ff9f0a", "width": 2},
                                    "thickness": 0.8,
                                    "value": THRESHOLD * 100,
                                },
                            },
                        ))
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=320,
                            margin=dict(t=40, b=20, l=40, r=40),
                            font={"family": "-apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif"},
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # ── Result card ────────────────────────────
                        result_bg = "rgba(255,69,58,0.1)" if is_cme else "rgba(48,209,88,0.1)"
                        result_border = "rgba(255,69,58,0.25)" if is_cme else "rgba(48,209,88,0.25)"
                        result_accent = "#ff453a" if is_cme else "#30d158"
                        result_icon = "⚠️" if is_cme else "✓"
                        result_label = "Alert — CME Detected" if is_cme else "Clear — Normal Conditions"
                        result_sub = "Space weather alert protocols should be activated immediately." if is_cme else "Solar wind parameters are within normal operating ranges."

                        st.markdown(f"""
                        <div style="
                            background: {result_bg};
                            border: 1px solid {result_border};
                            border-radius: 20px;
                            padding: 1.5rem 1.75rem;
                            margin-bottom: 1rem;
                        ">
                            <div style="display:flex; align-items:flex-start; gap:1rem;">
                                <div style="
                                    width:40px; height:40px;
                                    background: {result_accent};
                                    border-radius: 50%;
                                    display:flex; align-items:center; justify-content:center;
                                    color: white; font-size:1.1rem; font-weight:600;
                                    flex-shrink:0;
                                ">{result_icon}</div>
                                <div>
                                    <p style="font-size:1.0625rem; font-weight:600; color:#f5f5f7 !important; margin:0 0 4px 0;">{result_label}</p>
                                    <p style="font-size:0.875rem; color:#98989d !important; margin:0;">{result_sub}</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        col_p, col_t = st.columns(2)
                        with col_p:
                            st.metric("Probability Score", f"{prob * 100:.2f}%")
                        with col_t:
                            st.metric("Decision Threshold", f"{THRESHOLD * 100:.0f}%")

                    with tab3:
                        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                        st.markdown("""
                        <p style="font-size:0.8125rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.75rem;">Extracted Features</p>
                        """, unsafe_allow_html=True)
                        st.dataframe(
                            features_df,
                            use_container_width=True,
                        )

                        feature_names = features_df.columns.tolist()
                        feature_values = features_df.iloc[0].tolist()

                        fig_f = go.Figure(data=[go.Bar(
                            x=feature_names,
                            y=feature_values,
                            marker=dict(
                                color=feature_values,
                                colorscale=[[0, "#30d158"], [0.5, "#ff9f0a"], [1, "#ff453a"]],
                                showscale=False,
                                line=dict(width=0),
                            ),
                            text=[f"{v:.4f}" for v in feature_values],
                            textposition="outside",
                            textfont=dict(size=11, color="#98989d"),
                        )])
                        fig_f.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#98989d"), zeroline=False, linecolor="rgba(255,255,255,0.08)"),
                            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=11, color="#98989d"), zeroline=False),
                            height=360,
                            margin=dict(t=20, b=20, l=20, r=20),
                            font={"family": "-apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif"},
                            bargap=0.35,
                        )
                        st.plotly_chart(fig_f, use_container_width=True)

                        with st.expander("Feature Descriptions"):
                            st.markdown("""
| Feature | Scientific Significance |
|---|---|
| **Alpha-Proton Ratio** | Higher ratios indicate CME ejecta composition |
| **Speed Variability** | Increased turbulence during CME passage |
| **Alpha/VpStd Index** | Identifies alpha-rich coherent plasma structures |
| **Alpha/Temperature Ratio** | Detects cool, dense CME material |
""")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Ensure your CSV contains the required columns with valid numerical data.")

    else:
        # ── Empty state ───────────────────────────────
        st.markdown(f"""
        <div style="
            background: #1c1c1e;
            border: 1.5px dashed rgba(255,255,255,0.1);
            border-radius: 24px;
            padding: 4rem 2rem;
            text-align: center;
            margin-top: 1rem;
        ">
            <div style="font-size:3rem; margin-bottom:1.25rem; opacity:0.4;">☀️</div>
            <p style="font-size:1.125rem; font-weight:500; color:#f5f5f7 !important; margin:0 0 0.5rem 0;">No data uploaded yet</p>
            <p style="font-size:0.9375rem; color:#98989d !important; margin:0; max-width:380px; margin-left:auto; margin-right:auto; line-height:1.55;">
                Upload a CSV containing <strong style="color:#f5f5f7;">proton_density</strong>,
                <strong style="color:#f5f5f7;">proton_speed</strong>,
                <strong style="color:#f5f5f7;">proton_temperature</strong>, and
                <strong style="color:#f5f5f7;">alpha_density</strong> to begin analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# 🧠  HOW IT WORKS PAGE
# =====================================================
elif page == "How It Works":

    st.markdown("""
    <div style="margin-bottom: 2.5rem;">
        <p style="font-size:0.8125rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Methodology</p>
        <h1 style="margin:0; color:#f5f5f7 !important;">How It Works</h1>
    </div>
    """, unsafe_allow_html=True)

    workflow_tab, physics_tab, performance_tab = st.tabs(["  Pipeline  ", "  Physics Features  ", "  Performance  "])

    # ── Pipeline ──────────────────────────────────────
    with workflow_tab:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        steps = [
            ("01", "Data Acquisition", "Aditya-L1 SWIS L2 plasma data at 5-minute cadence. Parameters: Np, Vp, Tp, α.", "#0a84ff", "rgba(10,132,255,0.08)"),
            ("02", "Event Labeling", "CACTUS CME Catalog provides ground-truth Halo CME timestamps with ±1–2 day windows.", "#ff9f0a", "rgba(255,159,10,0.08)"),
            ("03", "Feature Engineering", "Physics-informed rolling statistics and ratio-based plasma composition markers.", "#30d158", "rgba(48,209,88,0.08)"),
            ("04", "ML Classification", "Soft voting ensemble (RF + XGBoost + LogReg) with probability-based output.", "#ff453a", "rgba(255,69,58,0.08)"),
        ]

        for num, title, desc, accent, bg in steps:
            st.markdown(f"""
            <div style="
                background: {bg};
                border: 1px solid {accent}33;
                border-radius: 18px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 0.75rem;
                display: flex;
                gap: 1.25rem;
                align-items: flex-start;
            ">
                <div style="
                    font-size: 0.75rem;
                    font-weight: 600;
                    color: {accent};
                    background: {accent}20;
                    padding: 4px 10px;
                    border-radius: 980px;
                    letter-spacing: 0.04em;
                    flex-shrink: 0;
                    margin-top: 2px;
                ">{num}</div>
                <div>
                    <p style="font-size:1rem; font-weight:500; color:#f5f5f7 !important; margin:0 0 4px 0;">{title}</p>
                    <p style="font-size:0.875rem; color:#98989d !important; margin:0; line-height:1.55;">{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        with st.expander("Full Pipeline Detail"):
            st.markdown("""
**Raw Data Collection**
- Aditya-L1 SWIS L2 Data: Proton density (Np), speed (Vp), temperature (Tp), Alpha particle density (α)
- CACTUS CME Catalog: Halo CME event timestamps and classifications
- Continuous 5-minute cadence measurements

**Event Window Extraction**
- Halo CME timestamps from CACTUS catalog
- Extract time windows: −1 to +2 days around events
- Balanced dataset: 10 CME + 30 Non-CME windows

**Feature Computation**
- Rolling statistics (1-hour windows)
- Ratio-based plasma composition markers
- Turbulence and variability indicators

**Model Inference**
- Soft voting ensemble (RF + XGBoost + LogReg)
- Probability-based classification
- Threshold optimization for high recall
""")

    # ── Physics Features ──────────────────────────────
    with physics_tab:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        features = [
            {
                "name": "Alpha-Proton Ratio",
                "formula": "α ÷ Np",
                "icon": "⚛",
                "accent": "#0a84ff",
                "bg": "rgba(10,132,255,0.07)",
                "insight": "CME ejecta typically shows enhanced alpha particle abundance compared to normal solar wind. Higher ratios are strong CME indicators.",
            },
            {
                "name": "Speed Variability",
                "formula": "rolling_std(Vp, 1hr)",
                "icon": "≋",
                "accent": "#ff9f0a",
                "bg": "rgba(255,159,10,0.07)",
                "insight": "CME-driven shocks create turbulence and speed fluctuations. Increased variability signals disturbed plasma conditions.",
            },
            {
                "name": "Alpha / VpStd Index",
                "formula": "(α/Np) ÷ VpStd",
                "icon": "◎",
                "accent": "#30d158",
                "bg": "rgba(48,209,88,0.07)",
                "insight": "Combines composition and coherence. High values indicate alpha-rich, low-turbulence CME ejecta cores.",
            },
            {
                "name": "Alpha / Temperature Ratio",
                "formula": "α ÷ Tp",
                "icon": "Δ",
                "accent": "#ff453a",
                "bg": "rgba(255,69,58,0.07)",
                "insight": "CME material is often cooler and denser than ambient solar wind. This ratio identifies thermodynamic signatures.",
            },
        ]

        col_a, col_b = st.columns(2)
        for i, f in enumerate(features):
            col = col_a if i % 2 == 0 else col_b
            with col:
                st.markdown(f"""
                <div style="
                    background: {f['bg']};
                    border: 1px solid {f['accent']}33;
                    border-radius: 20px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                ">
                    <div style="display:flex; align-items:center; gap:12px; margin-bottom:0.75rem;">
                        <div style="
                            width:36px; height:36px;
                            background:{f['accent']}20;
                            color:{f['accent']};
                            border-radius:10px;
                            display:flex; align-items:center; justify-content:center;
                            font-size:1rem; font-weight:600;
                        ">{f['icon']}</div>
                        <div>
                            <p style="font-size:0.9375rem; font-weight:500; color:#f5f5f7 !important; margin:0;">{f['name']}</p>
                            <code style="font-size:0.75rem; color:{f['accent']}; background:{f['accent']}18; padding:1px 8px; border-radius:4px;">{f['formula']}</code>
                        </div>
                    </div>
                    <p style="font-size:0.875rem; color:#98989d !important; margin:0; line-height:1.6;">{f['insight']}</p>
                </div>
                """, unsafe_allow_html=True)

    # ── Performance ───────────────────────────────────
    with performance_tab:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Accuracy", "85%", delta="Balanced")
        with c2:
            st.metric("Precision (CME)", "67%", delta="Conservative")
        with c3:
            st.metric("Recall (CME)", "100%", delta="Perfect")
        with c4:
            st.metric("F1-Score (CME)", "80%", delta="Optimised")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("""<p style="font-size:0.8125rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.75rem;">Ensemble Architecture</p>""", unsafe_allow_html=True)

        arch = [
            ("Random Forest", "Balanced class weights · robust to outliers · feature importance ranking", "#0a84ff", "rgba(10,132,255,0.07)"),
            ("XGBoost", "scale_pos_weight · gradient boosting · systematic bias reduction", "#ff9f0a", "rgba(255,159,10,0.07)"),
            ("Logistic Regression", "Interpretable linear boundary · probability calibration · baseline", "#30d158", "rgba(48,209,88,0.07)"),
        ]

        cols = st.columns(3)
        for col, (name, desc, accent, bg) in zip(cols, arch):
            with col:
                st.markdown(f"""
                <div style="
                    background:{bg};
                    border:1px solid {accent}33;
                    border-radius:18px;
                    padding:1.25rem 1.25rem;
                ">
                    <p style="font-size:0.9375rem; font-weight:500; color:#f5f5f7 !important; margin:0 0 6px 0;">{name}</p>
                    <p style="font-size:0.8125rem; color:#98989d !important; margin:0; line-height:1.55;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        col_adv, col_fut = st.columns(2)
        with col_adv:
            st.success("""
**Advantages**

No imaging required — enables real-time capability. Physics-informed features are scientifically interpretable. Lightweight architecture is suitable for spacecraft edge computing. Early warning detection is possible before impact.
""")
        with col_fut:
            st.info("""
**Future Enhancements**

Shock region and sheath detection. Multi-year SWIS dataset expansion. Real-time onboard inference via ONNX. Integration with operational forecasting models.
""")


# =====================================================
# 📊  ABOUT THE MODEL PAGE
# =====================================================
elif page == "About the Model":

    st.markdown("""
    <div style="margin-bottom: 2.5rem;">
        <p style="font-size:0.8125rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Background</p>
        <h1 style="margin:0; color:#f5f5f7 !important;">About the Model</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:#1c1c1e;
        border-radius:20px;
        border:1px solid rgba(255,255,255,0.06);
        padding:1.75rem 2rem;
        margin-bottom:1.5rem;
    ">
        <p style="font-size:0.8125rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.06em; margin:0 0 0.75rem 0;">Overview</p>
        <p style="font-size:1.0625rem; color:#f5f5f7 !important; margin:0 0 1rem 0; line-height:1.6; font-weight:400;">
            A novel approach to Coronal Mass Ejection detection using machine learning applied to
            in-situ plasma measurements — rather than traditional coronagraph imaging.
        </p>
        <p style="font-size:0.9375rem; color:#98989d !important; margin:0; line-height:1.65;">
            The system detects Halo-type CMEs from solar wind plasma data collected by the Aditya-L1
            spacecraft's Solar Wind Ion Spectrometer (SWIS) instrument, validated against the CACTUS CME catalog.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(apple_card("""
        <p style="font-size:0.8125rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.06em; margin:0 0 1rem 0;">Dataset Details</p>
        <div style="display:flex; flex-direction:column; gap:0.6rem;">
            <div>
                <p style="font-size:0.8125rem; color:#98989d !important; margin:0 0 2px 0;">Primary Sources</p>
                <p style="font-size:0.9375rem; color:#f5f5f7 !important; margin:0; font-weight:400;">Aditya-L1 SWIS L2 · CACTUS CME Catalog</p>
            </div>
            <div style="height:1px; background:rgba(255,255,255,0.06);"></div>
            <div>
                <p style="font-size:0.8125rem; color:#98989d !important; margin:0 0 2px 0;">Time Resolution</p>
                <p style="font-size:0.9375rem; color:#f5f5f7 !important; margin:0;">5 minutes</p>
            </div>
            <div style="height:1px; background:rgba(255,255,255,0.06);"></div>
            <div>
                <p style="font-size:0.8125rem; color:#98989d !important; margin:0 0 2px 0;">CME Events</p>
                <p style="font-size:0.9375rem; color:#f5f5f7 !important; margin:0;">10 Halo CMEs</p>
            </div>
            <div style="height:1px; background:rgba(255,255,255,0.06);"></div>
            <div>
                <p style="font-size:0.8125rem; color:#98989d !important; margin:0 0 2px 0;">Non-CME Windows</p>
                <p style="font-size:0.9375rem; color:#f5f5f7 !important; margin:0;">30 quiet periods</p>
            </div>
            <div style="height:1px; background:rgba(255,255,255,0.06);"></div>
            <div>
                <p style="font-size:0.8125rem; color:#98989d !important; margin:0 0 2px 0;">Window Size</p>
                <p style="font-size:0.9375rem; color:#f5f5f7 !important; margin:0;">−1 to +2 days around events</p>
            </div>
        </div>
        """), unsafe_allow_html=True)

        st.markdown(apple_card(f"""
        <p style="font-size:0.8125rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.06em; margin:0 0 1rem 0;">Model Parameters</p>
        <div style="display:flex; flex-direction:column; gap:0.6rem;">
            <div>
                <p style="font-size:0.8125rem; color:#98989d !important; margin:0 0 2px 0;">Algorithm</p>
                <p style="font-size:0.9375rem; color:#f5f5f7 !important; margin:0;">Soft Voting Ensemble</p>
            </div>
            <div style="height:1px; background:rgba(255,255,255,0.06);"></div>
            <div>
                <p style="font-size:0.8125rem; color:#98989d !important; margin:0 0 2px 0;">Components</p>
                <p style="font-size:0.9375rem; color:#f5f5f7 !important; margin:0;">RF + XGBoost + LogReg</p>
            </div>
            <div style="height:1px; background:rgba(255,255,255,0.06);"></div>
            <div>
                <p style="font-size:0.8125rem; color:#98989d !important; margin:0 0 2px 0;">Decision Threshold</p>
                <p style="font-size:0.9375rem; color:#f5f5f7 !important; margin:0;">{THRESHOLD}</p>
            </div>
            <div style="height:1px; background:rgba(255,255,255,0.06);"></div>
            <div>
                <p style="font-size:0.8125rem; color:#98989d !important; margin:0 0 2px 0;">Optimisation Goal</p>
                <p style="font-size:0.9375rem; color:#f5f5f7 !important; margin:0;">Maximise recall · minimise false negatives</p>
            </div>
        </div>
        """), unsafe_allow_html=True)

    with col2:
        st.markdown(apple_card("""
        <p style="font-size:0.8125rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.06em; margin:0 0 1rem 0;">Scientific Context</p>
        <p style="font-size:0.9375rem; font-weight:500; color:#f5f5f7 !important; margin:0 0 0.5rem 0;">What are Halo CMEs?</p>
        <p style="font-size:0.875rem; color:#98989d !important; margin:0 0 1rem 0; line-height:1.65;">
            Coronal Mass Ejections that appear to surround the Sun in coronagraph images, indicating
            they're directed toward or away from Earth. These are the most geo-effective solar events.
        </p>
        <div style="display:flex; flex-direction:column; gap:6px; margin-bottom:1rem;">
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:5px; height:5px; border-radius:50%; background:#ff453a; flex-shrink:0;"></div>
                <p style="font-size:0.875rem; color:#f5f5f7 !important; margin:0;">Satellite operations disruption</p>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:5px; height:5px; border-radius:50%; background:#ff9f0a; flex-shrink:0;"></div>
                <p style="font-size:0.875rem; color:#f5f5f7 !important; margin:0;">GPS navigation interference</p>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:5px; height:5px; border-radius:50%; background:#ff9f0a; flex-shrink:0;"></div>
                <p style="font-size:0.875rem; color:#f5f5f7 !important; margin:0;">Geomagnetic storms</p>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:5px; height:5px; border-radius:50%; background:#ff453a; flex-shrink:0;"></div>
                <p style="font-size:0.875rem; color:#f5f5f7 !important; margin:0;">Power grid infrastructure impact</p>
            </div>
        </div>
        <div style="height:1px; background:rgba(255,255,255,0.06); margin-bottom:1rem;"></div>
        <p style="font-size:0.9375rem; font-weight:500; color:#f5f5f7 !important; margin:0 0 0.5rem 0;">Why This Approach?</p>
        <p style="font-size:0.875rem; color:#98989d !important; margin:0; line-height:1.65;">
            Traditional detection relies on imaging instruments (LASCO, CACTUS), which introduce
            delays requiring ground processing. This ML system enables real-time onboard detection
            using only plasma measurements — enabling earlier warnings.
        </p>
        """), unsafe_allow_html=True)

    # ── Achievement stats ──────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""<p style="font-size:0.8125rem; font-weight:500; color:#98989d !important; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.75rem;">Key Results</p>""", unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("""
        <div style="background:rgba(48,209,88,0.1); border:1px solid rgba(48,209,88,0.25); border-radius:20px; padding:1.5rem; text-align:center;">
            <p style="font-size:2.5rem; font-weight:600; color:#30d158 !important; margin:0; letter-spacing:-0.03em;">0</p>
            <p style="font-size:0.875rem; font-weight:500; color:#f5f5f7 !important; margin:4px 0 2px 0;">False Negatives</p>
            <p style="font-size:0.8125rem; color:#98989d !important; margin:0;">Critical for safety</p>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div style="background:rgba(10,132,255,0.1); border:1px solid rgba(10,132,255,0.2); border-radius:20px; padding:1.5rem; text-align:center;">
            <p style="font-size:2.5rem; font-weight:600; color:#0a84ff !important; margin:0; letter-spacing:-0.03em;">4</p>
            <p style="font-size:0.875rem; font-weight:500; color:#f5f5f7 !important; margin:4px 0 2px 0;">Physics Features</p>
            <p style="font-size:0.8125rem; color:#98989d !important; margin:0;">Science-informed</p>
        </div>
        """, unsafe_allow_html=True)
    with a3:
        st.markdown("""
        <div style="background:rgba(255,159,10,0.1); border:1px solid rgba(255,159,10,0.2); border-radius:20px; padding:1.5rem; text-align:center;">
            <p style="font-size:2.5rem; font-weight:600; color:#ff9f0a !important; margin:0; letter-spacing:-0.03em;">85%</p>
            <p style="font-size:0.875rem; font-weight:500; color:#f5f5f7 !important; margin:4px 0 2px 0;">Accuracy</p>
            <p style="font-size:0.8125rem; color:#98989d !important; margin:0;">Cross-validated</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    st.success("""
**Academic Contribution** — This work demonstrates that physics-informed machine learning can match or exceed traditional detection methods while enabling real-time, onboard processing capabilities crucial for future space weather forecasting systems.
""")