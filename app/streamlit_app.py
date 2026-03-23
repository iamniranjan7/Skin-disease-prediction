"""
Skin Disease Prediction — Streamlit App
SVM (RBF Kernel) | UCI Dermatology Dataset | 97.3% Accuracy

Run:    streamlit run app/streamlit_app.py
Deploy: https://share.streamlit.io
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Auto-train if models don't exist ──────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def ensure_models():
    if not os.path.exists(os.path.join(MODEL_DIR, "svm_model.pkl")):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        data_path = os.path.join(BASE_DIR, "data", "dataset_35_dermatology__1_.csv")
        df = pd.read_csv(data_path)
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["Age"] = df["Age"].fillna(df["Age"].median())
        feature_cols = [c for c in df.columns if c != "class"]
        X = df[feature_cols].values
        y = df["class"].values
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_imp)
        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_sc, y)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model,   os.path.join(MODEL_DIR, "svm_model.pkl"))
        joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.pkl"))
        joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))

ensure_models()

st.set_page_config(
    page_title="SkinDx — Skin Disease Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_artifacts():
    m = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
    s = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    i = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    return m, s, i

model, scaler, imputer = load_artifacts()

CLASS_NAMES  = {1:"Psoriasis",2:"Seboreic Dermatitis",3:"Lichen Planus",4:"Pityriasis Rosea",5:"Chronic Dermatitis",6:"Pityriasis Rubra Pilaris"}
CLASS_COLORS = {1:"#3498DB",2:"#E74C3C",3:"#2ECC71",4:"#F39C12",5:"#9B59B6",6:"#1ABC9C"}
CLASS_EMOJIS = {1:"🔵",2:"🔴",3:"🟢",4:"🟡",5:"🟣",6:"🩵"}

FEATURE_COLS = [
    "erythema","scaling","definite_borders","itching","koebner_phenomenon",
    "polygonal_papules","follicular_papules","oral_mucosal_involvement",
    "knee_and_elbow_involvement","scalp_involvement","family_history",
    "melanin_incontinence","eosinophils_in_the_infiltrate","PNL_infiltrate",
    "fibrosis_of_the_papillary_dermis","exocytosis","acanthosis","hyperkeratosis",
    "parakeratosis","clubbing_of_the_rete_ridges","elongation_of_the_rete_ridges",
    "thinning_of_the_suprapapillary_epidermis","spongiform_pustule","munro_microabcess",
    "focal_hypergranulosis","disappearance_of_the_granular_layer",
    "vacuolisation_and_damage_of_basal_layer","spongiosis",
    "saw-tooth_appearance_of_retes","follicular_horn_plug",
    "perifollicular_parakeratosis","inflammatory_monoluclear_inflitrate",
    "band-like_infiltrate","Age"
]
CLINICAL  = FEATURE_COLS[:11]
HISTOPATH = FEATURE_COLS[11:33]

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #F0F4F8 !important; }
.block-container { padding-top: 0.5rem !important; padding-left: 1.5rem !important; padding-right: 1.5rem !important; }

.topnav { background:#0B1F3A; padding:0.9rem 1.5rem; margin:-0.5rem -1.5rem 1.5rem -1.5rem; display:flex; align-items:center; justify-content:space-between; border-bottom:3px solid #0D7377; }
.topnav-brand { font-family:'Space Mono',monospace; color:white; font-size:0.88rem; letter-spacing:1px; }
.topnav-brand span { color:#14A085; }
.topnav-badge { background:#0D7377; color:white; padding:0.2rem 0.8rem; border-radius:20px; font-family:'Space Mono',monospace; font-size:0.72rem; }

[data-testid="stSidebar"] { background:white !important; border-right:1px solid #CBD5E1 !important; }
.sidebar-hdr { background:#0B1F3A; padding:1rem 1rem 0.8rem; margin:-1rem -1rem 0.5rem -1rem; }
.sidebar-hdr h3 { color:white !important; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin:0 !important; }
.sidebar-hdr p  { color:#94A3B8; font-size:0.72rem; margin:0.2rem 0 0 !important; }
.sec-lbl { background:#F0F4F8; padding:0.35rem 0.5rem; margin:0.4rem -1rem; font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:#64748B; border-top:1px solid #CBD5E1; border-bottom:1px solid #CBD5E1; }
.sec-lbl-h { background:#F8F0FF; color:#8E44AD; border-color:#E8D5FF; }

.stButton > button { background:#0D7377 !important; color:white !important; border:none !important; border-radius:8px !important; font-weight:600 !important; font-family:'DM Sans',sans-serif !important; font-size:0.85rem !important; width:100%; transition:all 0.2s !important; }
.stButton > button:hover { background:#14A085 !important; }

.result-card { background:white; border-radius:14px; overflow:hidden; box-shadow:0 4px 24px rgba(11,31,58,0.08); margin-bottom:1.5rem; }
.result-card-hdr { padding:1rem 1.5rem; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #CBD5E1; }
.result-card-hdr h4 { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:#64748B; margin:0; }
.result-card-body { padding:1.5rem; }

.disease-row { display:flex; align-items:center; gap:1.2rem; margin-bottom:1.5rem; }
.disease-icon { width:68px; height:68px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:2rem; flex-shrink:0; }
.disease-name { font-size:1.55rem; font-weight:700; color:#0B1F3A; margin:0; }
.disease-badge { display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.7rem; color:white; font-weight:600; margin-top:0.25rem; font-family:'Space Mono',monospace; }

.conf-title { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:#64748B; margin-bottom:0.7rem; }
.conf-row { display:flex; align-items:center; gap:0.7rem; margin-bottom:0.5rem; }
.conf-lbl { font-size:0.76rem; color:#0B1F3A; width:180px; flex-shrink:0; }
.conf-track { flex:1; background:#F0F4F8; border-radius:100px; height:9px; overflow:hidden; }
.conf-fill  { height:100%; border-radius:100px; }
.conf-pct   { font-family:'Space Mono',monospace; font-size:0.68rem; color:#64748B; width:38px; text-align:right; }

.waiting { text-align:center; padding:3rem 2rem; color:#64748B; }
.waiting .wi { font-size:3rem; margin-bottom:1rem; }
.waiting .wt { font-size:0.88rem; }
.waiting .ws { font-size:0.75rem; color:#94A3B8; margin-top:0.4rem; }

.info-card { background:white; border-radius:12px; padding:1.1rem 1.3rem; box-shadow:0 2px 12px rgba(11,31,58,0.06); margin-bottom:1rem; }
.ic-title { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:#64748B; margin-bottom:0.6rem; }
.ic-row { display:flex; justify-content:space-between; align-items:center; padding:0.35rem 0; border-bottom:1px solid #F1F5F9; font-size:0.78rem; color:#0B1F3A; }
.ic-row:last-child { border:none; }
.ic-val { font-family:'Space Mono',monospace; font-weight:700; color:#0D7377; font-size:0.72rem; }
</style>
""", unsafe_allow_html=True)

# ── Navbar ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topnav">
  <div class="topnav-brand">🧬 <span>SkinDx</span> — SVM Classifier v1.0</div>
  <div class="topnav-badge">SVM Accuracy: 97.3%</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-hdr"><h3>Patient Feature Input</h3><p>Score: 0 = absent → 3 = severe</p></div>', unsafe_allow_html=True)

    fv = {}

    st.markdown('<div class="sec-lbl">🩺 Clinical Features (11)</div>', unsafe_allow_html=True)
    for feat in CLINICAL:
        label = feat.replace("_"," ").title()
        if feat == "family_history":
            fv[feat] = st.selectbox("Family History", [0,1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
        else:
            fv[feat] = st.select_slider(label, options=[0,1,2,3], value=0)

    st.markdown('<div class="sec-lbl sec-lbl-h">🔬 Histopathological Features (22)</div>', unsafe_allow_html=True)
    for feat in HISTOPATH:
        label = feat.replace("_"," ").replace("-"," ").title()
        fv[feat] = st.select_slider(label, options=[0,1,2,3], value=0)

    st.markdown('<div class="sec-lbl">📊 Demographic</div>', unsafe_allow_html=True)
    fv["Age"] = st.number_input("Age (years)", min_value=0, max_value=120, value=35)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("▶  Run Prediction", use_container_width=True)
    if st.button("↺  Reset All", use_container_width=True):
        st.rerun()

# ── Main layout ────────────────────────────────────────────────────────────────
col_main, col_side = st.columns([3, 1], gap="large")

with col_side:
    st.markdown("""
    <div class="info-card">
        <div class="ic-title">Model Performance</div>
        <div class="ic-row"><span>Algorithm</span><span class="ic-val">SVM (RBF)</span></div>
        <div class="ic-row"><span>Test Accuracy</span><span class="ic-val">97.30%</span></div>
        <div class="ic-row"><span>F1 Macro</span><span class="ic-val">96.97%</span></div>
        <div class="ic-row"><span>CV Score</span><span class="ic-val">97.60%</span></div>
        <div class="ic-row"><span>Train Samples</span><span class="ic-val">292</span></div>
        <div class="ic-row"><span>Test Samples</span><span class="ic-val">74</span></div>
    </div>
    <div class="info-card">
        <div class="ic-title">Dataset Info</div>
        <div class="ic-row"><span>Records</span><span class="ic-val">366</span></div>
        <div class="ic-row"><span>Features</span><span class="ic-val">34</span></div>
        <div class="ic-row"><span>Classes</span><span class="ic-val">6</span></div>
        <div class="ic-row"><span>Missing</span><span class="ic-val">8 (imputed)</span></div>
        <div class="ic-row"><span>Scaling</span><span class="ic-val">StandardScaler</span></div>
        <div class="ic-row"><span>Source</span><span class="ic-val">UCI Derm.</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_main:
    if not predict_btn:
        st.markdown("""
        <div class="result-card">
            <div class="result-card-hdr"><h4>Prediction Result</h4></div>
            <div class="result-card-body">
                <div class="waiting">
                    <div class="wi">🧬</div>
                    <div class="wt">Set patient feature values on the left, then click <strong>Run Prediction</strong></div>
                    <div class="ws">SVM (RBF Kernel) · 97.3% accuracy · 6 disease classes</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        features   = [fv[f] for f in FEATURE_COLS]
        X          = np.array(features).reshape(1, -1)
        X_imp      = imputer.transform(X)
        X_scaled   = scaler.transform(X_imp)
        pred_class = int(model.predict(X_scaled)[0])
        pred_proba = model.predict_proba(X_scaled)[0]
        classes    = model.classes_
        confidence = float(max(pred_proba))
        disease    = CLASS_NAMES[pred_class]
        color      = CLASS_COLORS[pred_class]
        emoji      = CLASS_EMOJIS[pred_class]
        proba_dict = {int(c): float(p) for c, p in zip(classes, pred_proba)}

        conf_bars = "".join([
            f'<div class="conf-row">'
            f'<div class="conf-lbl">{CLASS_NAMES[c]}</div>'
            f'<div class="conf-track"><div class="conf-fill" style="width:{proba_dict.get(c,0)*100:.1f}%;background:{CLASS_COLORS[c]};"></div></div>'
            f'<div class="conf-pct">{proba_dict.get(c,0)*100:.1f}%</div>'
            f'</div>'
            for c in sorted(proba_dict, key=lambda x: proba_dict[x], reverse=True)
        ])

        st.markdown(f"""
        <div class="result-card">
            <div class="result-card-hdr"><h4>Prediction Result</h4></div>
            <div class="result-card-body">
                <div class="disease-row">
                    <div class="disease-icon" style="background:{color}22;">{emoji}</div>
                    <div>
                        <div class="disease-name">{disease}</div>
                        <span class="disease-badge" style="background:{color};">Class {pred_class} · {confidence:.1%} confidence</span>
                    </div>
                </div>
                <div class="conf-title">Similarity Score — All 6 Classes</div>
                {conf_bars}
            </div>
        </div>
        """, unsafe_allow_html=True)

        active = {f: int(v) for f, v in fv.items() if f != "Age" and int(v) > 0}
        if active:
            feat_df = pd.DataFrame([
                (f.replace("_"," ").replace("-"," ").title(), v)
                for f, v in sorted(active.items(), key=lambda x: -x[1])
            ], columns=["Feature","Score"])

            fig = go.Figure(go.Bar(
                x=feat_df["Score"], y=feat_df["Feature"], orientation="h",
                marker=dict(color=feat_df["Score"], colorscale=[[0,"#D4F5F0"],[0.5,"#0D9488"],[1,"#0B1F3A"]], cmin=0, cmax=3),
                text=feat_df["Score"], textposition="outside"
            ))
            fig.update_layout(
                title=dict(text="Active Feature Values (score > 0)", font=dict(size=11, color="#64748B")),
                height=max(200, len(active)*30),
                margin=dict(l=0, r=40, t=30, b=10),
                xaxis=dict(range=[0,3.8], showgrid=False, zeroline=False, tickvals=[0,1,2,3]),
                yaxis=dict(tickfont=dict(size=9)),
                plot_bgcolor="white", paper_bgcolor="white"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Class distribution chart
    cc = {"Psoriasis":112,"Seboreic Dermatitis":61,"Lichen Planus":72,"Pityriasis Rosea":49,"Chronic Dermatitis":52,"Pityriasis Rubra Pilaris":20}
    fig2 = go.Figure(go.Bar(
        x=list(cc.keys()), y=list(cc.values()),
        marker_color=list(CLASS_COLORS.values()),
        text=[f"{v} ({v/366*100:.1f}%)" for v in cc.values()],
        textposition="outside", opacity=0.85
    ))
    fig2.update_layout(
        title=dict(text="Dataset Class Distribution", font=dict(size=11, color="#64748B")),
        height=260, margin=dict(l=0, r=0, t=30, b=10),
        xaxis=dict(tickangle=-20, tickfont=dict(size=9)),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)
