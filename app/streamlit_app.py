"""
Skin Disease Prediction — Streamlit App
========================================
SVM (RBF Kernel) | UCI Dermatology Dataset | 97.3% Accuracy

Run:
    streamlit run app/streamlit_app.py

Deploy free:
    https://share.streamlit.io → connect your GitHub repo
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Disease Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_artifacts():
    model   = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
    scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    return model, scaler, imputer

model, scaler, imputer = load_artifacts()

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    1: "Psoriasis",
    2: "Seboreic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris"
}

CLASS_COLORS = {
    1: "#3498DB", 2: "#E74C3C", 3: "#2ECC71",
    4: "#F39C12", 5: "#9B59B6", 6: "#1ABC9C"
}

FEATURE_COLS = [
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history", "melanin_incontinence",
    "eosinophils_in_the_infiltrate", "PNL_infiltrate",
    "fibrosis_of_the_papillary_dermis", "exocytosis", "acanthosis",
    "hyperkeratosis", "parakeratosis", "clubbing_of_the_rete_ridges",
    "elongation_of_the_rete_ridges", "thinning_of_the_suprapapillary_epidermis",
    "spongiform_pustule", "munro_microabcess", "focal_hypergranulosis",
    "disappearance_of_the_granular_layer", "vacuolisation_and_damage_of_basal_layer",
    "spongiosis", "saw-tooth_appearance_of_retes", "follicular_horn_plug",
    "perifollicular_parakeratosis", "inflammatory_monoluclear_inflitrate",
    "band-like_infiltrate", "Age"
]

CLINICAL    = FEATURE_COLS[:11]
HISTOPATH   = FEATURE_COLS[11:33]

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stSlider > div > div { background: #0D7377 !important; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 4px solid #0D7377; margin-bottom: 0.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card h4 { color: #0B1F3A; font-size: 0.85rem; margin: 0; }
    .metric-card .val { color: #0D7377; font-size: 1.4rem; font-weight: 700; }
    .result-box {
        padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
        border: 2px solid; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🧬 Skin Disease Classifier")
st.markdown("**UCI Dermatology Dataset · SVM (RBF Kernel) · 97.3% Test Accuracy**")
st.markdown("---")

# ── Sidebar: Feature Inputs ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Patient Feature Input")
    st.caption("Score 0 = absent, 3 = most severe")

    feature_values = {}

    st.markdown("**── Clinical Features ──**")
    for feat in CLINICAL:
        label = feat.replace("_", " ").title()
        if feat == "family_history":
            feature_values[feat] = st.selectbox(label, [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        else:
            feature_values[feat] = st.slider(label, 0, 3, 0)

    st.markdown("**── Histopathological Features ──**")
    st.caption("🔬 From biopsy/microscopy analysis")
    for feat in HISTOPATH:
        label = feat.replace("_", " ").replace("-", " ").title()
        feature_values[feat] = st.slider(label, 0, 3, 0)

    st.markdown("**── Demographic ──**")
    feature_values["Age"] = st.number_input("Age (years)", 0, 120, 35)

    predict_btn = st.button("🔍 Run Prediction", type="primary", use_container_width=True)
    reset_btn   = st.button("↺ Reset All", use_container_width=True)

# ── Main Area ──────────────────────────────────────────────────────────────────
col_result, col_info = st.columns([3, 1])

with col_info:
    st.markdown("**Model Stats**")
    for label, val in [
        ("Algorithm", "SVM (RBF)"),
        ("Test Accuracy", "97.30%"),
        ("F1 Macro", "96.97%"),
        ("CV Score", "97.60% ± 0.009"),
        ("Features", "34"),
        ("Classes", "6"),
        ("Dataset", "366 records"),
    ]:
        st.markdown(f"""
        <div class="metric-card">
            <h4>{label}</h4>
            <div class="val">{val}</div>
        </div>""", unsafe_allow_html=True)

with col_result:
    if predict_btn:
        # Build feature array
        features = [feature_values[f] for f in FEATURE_COLS]
        X = np.array(features).reshape(1, -1)
        X_imp    = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)

        pred_class = int(model.predict(X_scaled)[0])
        pred_proba = model.predict_proba(X_scaled)[0]
        confidence = float(max(pred_proba))
        disease    = CLASS_NAMES[pred_class]
        color      = CLASS_COLORS[pred_class]

        # ── Result box ──
        st.markdown(f"""
        <div class="result-box" style="border-color:{color};background:{color}11;">
            <h2 style="color:{color};margin:0;">🔬 {disease}</h2>
            <p style="color:#64748B;margin:0.3rem 0 0;">Class {pred_class} · Confidence: {confidence:.1%}</p>
        </div>""", unsafe_allow_html=True)

        # ── Probability bar chart ──
        st.markdown("**Similarity Scores — All 6 Classes**")
        prob_df = pd.DataFrame({
            "Disease": [CLASS_NAMES[i] for i in range(1, 7)],
            "Probability": pred_proba,
            "Color": [CLASS_COLORS[i] for i in range(1, 7)]
        }).sort_values("Probability", ascending=True)

        fig = go.Figure(go.Bar(
            x=prob_df["Probability"],
            y=prob_df["Disease"],
            orientation="h",
            marker_color=prob_df["Color"],
            text=[f"{p:.1%}" for p in prob_df["Probability"]],
            textposition="outside"
        ))
        fig.update_layout(
            height=280, margin=dict(l=0, r=60, t=10, b=10),
            xaxis=dict(range=[0, 1.1], showgrid=False, zeroline=False),
            yaxis=dict(tickfont=dict(size=11)),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Top features ──
        st.markdown("**Patient Feature Summary**")
        active_features = {
            f: v for f, v in feature_values.items()
            if f != "Age" and v > 0
        }
        if active_features:
            feat_df = pd.DataFrame(
                [(k.replace("_"," ").title(), v) for k, v in active_features.items()],
                columns=["Feature", "Score"]
            ).sort_values("Score", ascending=False)
            fig2 = px.bar(
                feat_df, x="Score", y="Feature", orientation="h",
                color="Score", color_continuous_scale=["#D4F5F0", "#0D7377"],
                range_x=[0, 3]
            )
            fig2.update_layout(
                height=max(200, len(active_features) * 28),
                margin=dict(l=0, r=0, t=10, b=10),
                showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No features above 0. Set some feature values in the sidebar to see the chart.")

    else:
        st.info("👈 Set patient features in the sidebar, then click **Run Prediction**")

        # ── EDA summary when idle ──
        st.markdown("**Dataset Class Distribution**")
        class_counts = {"Psoriasis":112,"Seboreic Dermatitis":61,"Lichen Planus":72,
                        "Pityriasis Rosea":49,"Chronic Dermatitis":52,"Pityriasis Rubra Pilaris":20}
        df_cls = pd.DataFrame(list(class_counts.items()), columns=["Disease", "Count"])
        fig3 = px.bar(df_cls, x="Disease", y="Count",
                      color="Disease",
                      color_discrete_sequence=list(CLASS_COLORS.values()))
        fig3.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=60),
            showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_tickangle=-25
        )
        st.plotly_chart(fig3, use_container_width=True)
