"""
Skin Disease Prediction — Streamlit App (3-Tab Version)
SVM (RBF Kernel) | UCI Dermatology Dataset | 97.3% Accuracy
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Auto-train if models missing ───────────────────────────────────────────────
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
        fc = [c for c in df.columns if c != "class"]
        X  = df[fc].values
        y  = df["class"].values
        imp = SimpleImputer(strategy="median"); X_i = imp.fit_transform(X)
        sc  = StandardScaler();                 X_s = sc.fit_transform(X_i)
        m   = SVC(kernel="rbf", probability=True, random_state=42); m.fit(X_s, y)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(m,   os.path.join(MODEL_DIR, "svm_model.pkl"))
        joblib.dump(sc,  os.path.join(MODEL_DIR, "scaler.pkl"))
        joblib.dump(imp, os.path.join(MODEL_DIR, "imputer.pkl"))

ensure_models()

st.set_page_config(page_title="SkinDx — Skin Disease Predictor",
                   page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")

@st.cache_resource
def load_artifacts():
    m = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
    s = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    i = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    return m, s, i

model, scaler, imputer = load_artifacts()

CLASS_NAMES  = {1:"Psoriasis",2:"Seboreic Dermatitis",3:"Lichen Planus",
                4:"Pityriasis Rosea",5:"Chronic Dermatitis",6:"Pityriasis Rubra Pilaris"}
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

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:#F0F4F8!important;}
.block-container{padding-top:0!important;padding-left:1.5rem!important;padding-right:1.5rem!important;max-width:100%!important;}

/* Navbar */
.topnav{background:#0B1F3A;padding:0 1.5rem;display:flex;align-items:center;justify-content:space-between;height:54px;border-bottom:3px solid #0D7377;margin:-1rem -1.5rem 0 -1.5rem;}
.topnav-brand{font-family:'Space Mono',monospace;color:white;font-size:0.85rem;letter-spacing:1px;}
.topnav-brand span{color:#14A085;}
.topnav-badge{background:#0D7377;color:white;padding:0.2rem 0.8rem;border-radius:20px;font-family:'Space Mono',monospace;font-size:0.72rem;}

/* Sidebar */
[data-testid="stSidebar"]{background:white!important;border-right:1px solid #CBD5E1!important;}
.sb-hdr{background:#0B1F3A;padding:1rem;margin:-1rem -1rem 0.5rem -1rem;}
.sb-hdr h3{color:white!important;font-size:0.82rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin:0!important;}
.sb-hdr p{color:#94A3B8;font-size:0.7rem;margin:0.15rem 0 0!important;}
.sec-lbl{background:#F0F4F8;padding:0.32rem 0.5rem;margin:0.4rem -1rem;font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#64748B;border-top:1px solid #CBD5E1;border-bottom:1px solid #CBD5E1;}
.sec-lbl-h{background:#F8F0FF;color:#8E44AD;border-color:#E8D5FF;}

/* Buttons */
.stButton>button{background:#0D7377!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;font-family:'DM Sans',sans-serif!important;font-size:0.85rem!important;width:100%;transition:all 0.2s!important;}
.stButton>button:hover{background:#14A085!important;}

/* Result card */
.rcard{background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 24px rgba(11,31,58,0.08);margin-bottom:1.5rem;}
.rcard-hdr{padding:0.9rem 1.5rem;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #CBD5E1;}
.rcard-hdr h4{font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#64748B;margin:0;}
.rcard-body{padding:1.5rem;}
.disease-row{display:flex;align-items:center;gap:1.2rem;margin-bottom:1.5rem;}
.d-icon{width:66px;height:66px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.9rem;flex-shrink:0;}
.d-name{font-size:1.5rem;font-weight:700;color:#0B1F3A;margin:0;}
.d-badge{display:inline-block;padding:2px 12px;border-radius:20px;font-size:0.68rem;color:white;font-weight:600;margin-top:0.2rem;font-family:'Space Mono',monospace;}
.conf-ttl{font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#64748B;margin-bottom:0.65rem;}
.conf-row{display:flex;align-items:center;gap:0.65rem;margin-bottom:0.48rem;}
.conf-lbl{font-size:0.74rem;color:#0B1F3A;width:175px;flex-shrink:0;}
.conf-track{flex:1;background:#F0F4F8;border-radius:100px;height:9px;overflow:hidden;}
.conf-fill{height:100%;border-radius:100px;}
.conf-pct{font-family:'Space Mono',monospace;font-size:0.66rem;color:#64748B;width:36px;text-align:right;}
.waiting{text-align:center;padding:3rem 2rem;color:#64748B;}
.wi{font-size:3rem;margin-bottom:1rem;}
.wt{font-size:0.88rem;}
.ws{font-size:0.74rem;color:#94A3B8;margin-top:0.4rem;}

/* Info cards */
.icard{background:white;border-radius:12px;padding:1rem 1.2rem;box-shadow:0 2px 12px rgba(11,31,58,0.06);margin-bottom:1rem;}
.ict{font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#64748B;margin-bottom:0.55rem;}
.icr{display:flex;justify-content:space-between;align-items:center;padding:0.32rem 0;border-bottom:1px solid #F1F5F9;font-size:0.76rem;color:#0B1F3A;}
.icr:last-child{border:none;}
.icv{font-family:'Space Mono',monospace;font-weight:700;color:#0D7377;font-size:0.68rem;}

/* ── Tab 2: Compare cards ── */
.deploy-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;margin-bottom:1.5rem;}
.deploy-card{background:white;border-radius:14px;overflow:hidden;box-shadow:0 3px 16px rgba(11,31,58,0.07);border:2px solid transparent;transition:border-color 0.2s;}
.deploy-card:hover{border-color:#0D7377;}
.dc-hdr{padding:1.2rem 1.5rem;display:flex;align-items:center;gap:0.9rem;border-bottom:1px solid #f0f0f0;}
.dc-icon{width:44px;height:44px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;}
.dc-hdr h3{font-size:1rem;font-weight:700;margin:0;}
.dc-hdr p{font-size:0.74rem;color:#64748B;margin:0.1rem 0 0;}
.dc-body{padding:1.2rem 1.5rem;}
.pc-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-bottom:1rem;}
.pc-col h5{font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.4rem;}
.pros h5{color:#27AE60;} .cons h5{color:#E74C3C;}
.pc-item{font-size:0.76rem;color:#0B1F3A;padding:0.18rem 0;display:flex;gap:0.35rem;}
.pc-item::before{content:'✓';color:#27AE60;font-weight:700;flex-shrink:0;}
.cons .pc-item::before{content:'✗';color:#E74C3C;}
.uc-tag{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.68rem;font-weight:600;margin:2px;}

/* Matrix table */
.mtable{width:100%;font-size:0.78rem;border-collapse:collapse;}
.mtable th{padding:0.55rem 0.8rem;border-bottom:2px solid #CBD5E1;text-align:left;background:#F0F4F8;}
.mtable th:nth-child(2){color:#C0392B;text-align:center;}
.mtable th:nth-child(3){color:#166534;text-align:center;}
.mtable td{padding:0.48rem 0.8rem;border-bottom:1px solid #F1F5F9;}
.mtable td:nth-child(2){color:#C0392B;text-align:center;font-weight:600;}
.mtable td:nth-child(3){color:#166534;text-align:center;font-weight:600;}

/* Recommendation box */
.rec-box{background:linear-gradient(135deg,#0B1F3A,#1a3a5c);border-radius:14px;padding:1.5rem;color:white;display:flex;gap:1.2rem;align-items:flex-start;margin-top:1.2rem;}
.rec-icon{font-size:2.2rem;flex-shrink:0;}
.rec-box h3{font-size:1rem;font-weight:700;margin-bottom:0.4rem;}
.rec-box p{font-size:0.82rem;color:#D4F5F0;line-height:1.7;margin:0;}

/* Code block */
.code-tabs-wrap{border-bottom:2px solid #CBD5E1;display:flex;gap:0;margin-bottom:0;}
.code-pre{background:#0B1F3A;color:#E2E8F0;padding:1.2rem 1.5rem;border-radius:0 0 12px 12px;font-family:'Space Mono',monospace;font-size:0.7rem;line-height:1.8;overflow-x:auto;margin:0;}
.kw{color:#67D7EE;} .str{color:#A8D8A8;} .cm{color:#4a5568;} .fn{color:#F4A823;} .num{color:#FF9B71;}

/* ── Tab 3: Architecture ── */
.flow-wrap{display:flex;align-items:center;flex-wrap:wrap;gap:0;margin:1rem 0;}
.flow-step{background:#F0F4F8;border-radius:10px;padding:0.9rem 1rem;text-align:center;min-width:120px;}
.flow-step .fs-icon{font-size:1.4rem;margin-bottom:0.3rem;}
.flow-step .fs-lbl{font-size:0.68rem;font-weight:700;color:#0B1F3A;}
.flow-step .fs-sub{font-size:0.6rem;color:#64748B;margin-top:0.12rem;}
.flow-arrow{color:#0D7377;font-size:1.1rem;padding:0 0.4rem;}
.pipeline-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.1rem;margin:1.2rem 0;}
.pc{background:white;border-radius:10px;padding:1rem 1.1rem;box-shadow:0 2px 10px rgba(11,31,58,0.06);border-left:4px solid #0D7377;}
.pc h4{font-size:0.8rem;font-weight:700;color:#0B1F3A;margin-bottom:0.4rem;}
.pc p{font-size:0.74rem;color:#64748B;line-height:1.6;margin:0;}
.stack-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-top:0.75rem;}
.stack-item{text-align:center;padding:0.9rem;background:#F0F4F8;border-radius:8px;}
.stack-item .si{font-size:1.4rem;margin-bottom:0.3rem;}
.stack-item .sl{font-size:0.75rem;font-weight:700;color:#0B1F3A;}
.stack-item .ss{font-size:0.65rem;color:#64748B;margin-top:0.15rem;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="topnav">
  <div class="topnav-brand">🧬 <span>SkinDx</span> — SVM Classifier v1.0</div>
  <div class="topnav-badge">SVM Accuracy: 97.3%</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (only for predictor tab)
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-hdr"><h3>Patient Feature Input</h3><p>Score: 0 = absent → 3 = severe</p></div>', unsafe_allow_html=True)
    fv = {}
    st.markdown('<div class="sec-lbl">🩺 Clinical Features (11)</div>', unsafe_allow_html=True)
    for feat in CLINICAL:
        label = feat.replace("_"," ").title()
        if feat == "family_history":
            fv[feat] = st.selectbox("Family History",[0,1],format_func=lambda x:"No (0)" if x==0 else "Yes (1)")
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

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔬  Predictor", "⚖️  Flask vs Streamlit", "🏗️  Architecture"])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTOR
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    col_main, col_side = st.columns([3,1], gap="large")

    with col_side:
        st.markdown("""
        <div class="icard">
            <div class="ict">Model Performance</div>
            <div class="icr"><span>Algorithm</span><span class="icv">SVM (RBF)</span></div>
            <div class="icr"><span>Test Accuracy</span><span class="icv">97.30%</span></div>
            <div class="icr"><span>F1 Macro</span><span class="icv">96.97%</span></div>
            <div class="icr"><span>CV Score</span><span class="icv">97.60%</span></div>
            <div class="icr"><span>Train Samples</span><span class="icv">292</span></div>
            <div class="icr"><span>Test Samples</span><span class="icv">74</span></div>
        </div>
        <div class="icard">
            <div class="ict">Dataset Info</div>
            <div class="icr"><span>Records</span><span class="icv">366</span></div>
            <div class="icr"><span>Features</span><span class="icv">34</span></div>
            <div class="icr"><span>Classes</span><span class="icv">6</span></div>
            <div class="icr"><span>Missing</span><span class="icv">8 (imputed)</span></div>
            <div class="icr"><span>Scaling</span><span class="icv">StandardScaler</span></div>
            <div class="icr"><span>Source</span><span class="icv">UCI Derm.</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_main:
        if not predict_btn:
            st.markdown("""
            <div class="rcard">
                <div class="rcard-hdr"><h4>Prediction Result</h4></div>
                <div class="rcard-body">
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
            X          = np.array(features).reshape(1,-1)
            X_imp      = imputer.transform(X)
            X_scaled   = scaler.transform(X_imp)
            pred_class = int(model.predict(X_scaled)[0])
            pred_proba = model.predict_proba(X_scaled)[0]
            classes    = model.classes_
            confidence = float(max(pred_proba))
            disease    = CLASS_NAMES[pred_class]
            color      = CLASS_COLORS[pred_class]
            emoji      = CLASS_EMOJIS[pred_class]
            proba_dict = {int(c):float(p) for c,p in zip(classes,pred_proba)}

            conf_bars = "".join([
                f'<div class="conf-row"><div class="conf-lbl">{CLASS_NAMES[c]}</div>'
                f'<div class="conf-track"><div class="conf-fill" style="width:{proba_dict.get(c,0)*100:.1f}%;background:{CLASS_COLORS[c]};"></div></div>'
                f'<div class="conf-pct">{proba_dict.get(c,0)*100:.1f}%</div></div>'
                for c in sorted(proba_dict, key=lambda x: proba_dict[x], reverse=True)
            ])

            st.markdown(f"""
            <div class="rcard">
                <div class="rcard-hdr"><h4>Prediction Result</h4></div>
                <div class="rcard-body">
                    <div class="disease-row">
                        <div class="d-icon" style="background:{color}22;">{emoji}</div>
                        <div>
                            <div class="d-name">{disease}</div>
                            <span class="d-badge" style="background:{color};">Class {pred_class} · {confidence:.1%} confidence</span>
                        </div>
                    </div>
                    <div class="conf-ttl">Similarity Score — All 6 Classes</div>
                    {conf_bars}
                </div>
            </div>
            """, unsafe_allow_html=True)

            active = {f:int(v) for f,v in fv.items() if f!="Age" and int(v)>0}
            if active:
                feat_df = pd.DataFrame([(f.replace("_"," ").replace("-"," ").title(),v)
                          for f,v in sorted(active.items(),key=lambda x:-x[1])],columns=["Feature","Score"])
                fig = go.Figure(go.Bar(
                    x=feat_df["Score"], y=feat_df["Feature"], orientation="h",
                    marker=dict(color=feat_df["Score"],colorscale=[[0,"#D4F5F0"],[0.5,"#0D9488"],[1,"#0B1F3A"]],cmin=0,cmax=3),
                    text=feat_df["Score"], textposition="outside"
                ))
                fig.update_layout(title=dict(text="Active Feature Values",font=dict(size=11,color="#64748B")),
                    height=max(200,len(active)*30), margin=dict(l=0,r=40,t=30,b=10),
                    xaxis=dict(range=[0,3.8],showgrid=False,zeroline=False,tickvals=[0,1,2,3]),
                    yaxis=dict(tickfont=dict(size=9)),plot_bgcolor="white",paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)

        # Class distribution always visible
        cc = {"Psoriasis":112,"Seboreic Dermatitis":61,"Lichen Planus":72,
              "Pityriasis Rosea":49,"Chronic Dermatitis":52,"Pityriasis Rubra Pilaris":20}
        fig2 = go.Figure(go.Bar(
            x=list(cc.keys()), y=list(cc.values()),
            marker_color=list(CLASS_COLORS.values()),
            text=[f"{v} ({v/366*100:.1f}%)" for v in cc.values()],
            textposition="outside", opacity=0.85
        ))
        fig2.update_layout(
            title=dict(text="Dataset Class Distribution",font=dict(size=11,color="#64748B")),
            height=255, margin=dict(l=0,r=0,t=30,b=10),
            xaxis=dict(tickangle=-20,tickfont=dict(size=9)),
            yaxis=dict(showgrid=False,zeroline=False),
            plot_bgcolor="white",paper_bgcolor="white",showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — FLASK vs STREAMLIT
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Flask REST API vs Streamlit App")
    st.markdown("Two deployment options for the SVM Skin Disease Classifier — choose based on your use case.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Two deploy cards
    st.markdown("""
    <div class="deploy-grid">

      <div class="deploy-card">
        <div class="dc-hdr" style="background:#FFF5F5;border-bottom:1px solid #FFE0E0;">
          <div class="dc-icon" style="background:#FECACA;">🌶️</div>
          <div><h3 style="color:#C0392B;">Flask REST API</h3><p>Backend microservice with JSON endpoints</p></div>
        </div>
        <div class="dc-body">
          <div class="pc-grid">
            <div class="pros">
              <h5>✅ Pros</h5>
              <div class="pc-item">Full control over API design</div>
              <div class="pc-item">Integrates with any frontend</div>
              <div class="pc-item">Production-grade scalable</div>
              <div class="pc-item">Easy to add auth & rate limiting</div>
              <div class="pc-item">Lightweight & fast</div>
            </div>
            <div class="cons">
              <h5>❌ Cons</h5>
              <div class="pc-item">No built-in UI</div>
              <div class="pc-item">More code to write</div>
              <div class="pc-item">Separate frontend needed</div>
              <div class="pc-item">Harder for non-devs</div>
            </div>
          </div>
          <p style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#64748B;margin-bottom:0.4rem;">Best For</p>
          <span class="uc-tag" style="background:#FFF5F5;color:#C0392B;border:1px solid #FECACA;">Hospital EHR Integration</span>
          <span class="uc-tag" style="background:#FFF5F5;color:#C0392B;border:1px solid #FECACA;">Mobile App Backend</span>
          <span class="uc-tag" style="background:#FFF5F5;color:#C0392B;border:1px solid #FECACA;">Microservice Architecture</span>
          <div style="margin-top:0.9rem;padding:0.65rem;background:#FFF5F5;border-radius:8px;">
            <p style="font-size:0.75rem;color:#7F1D1D;margin:0;"><strong>Typical Setup:</strong> Flask + Gunicorn + Nginx on AWS EC2 or Google Cloud Run</p>
          </div>
        </div>
      </div>

      <div class="deploy-card">
        <div class="dc-hdr" style="background:#F0FDF4;border-bottom:1px solid #BBF7D0;">
          <div class="dc-icon" style="background:#BBF7D0;">⚡</div>
          <div><h3 style="color:#166534;">Streamlit App</h3><p>Interactive data science web application</p></div>
        </div>
        <div class="dc-body">
          <div class="pc-grid">
            <div class="pros">
              <h5>✅ Pros</h5>
              <div class="pc-item">Built-in beautiful UI</div>
              <div class="pc-item">Rapid prototyping</div>
              <div class="pc-item">Free deployment (Streamlit Cloud)</div>
              <div class="pc-item">Great for demos</div>
              <div class="pc-item">Built-in charts & widgets</div>
            </div>
            <div class="cons">
              <h5>❌ Cons</h5>
              <div class="pc-item">Not a true REST API</div>
              <div class="pc-item">Limited customisation</div>
              <div class="pc-item">Restarts on each interaction</div>
              <div class="pc-item">Not ideal for high traffic</div>
            </div>
          </div>
          <p style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#64748B;margin-bottom:0.4rem;">Best For</p>
          <span class="uc-tag" style="background:#F0FDF4;color:#166534;border:1px solid #BBF7D0;">Capstone Demo</span>
          <span class="uc-tag" style="background:#F0FDF4;color:#166534;border:1px solid #BBF7D0;">Doctor-facing Prototype</span>
          <span class="uc-tag" style="background:#F0FDF4;color:#166534;border:1px solid #BBF7D0;">Portfolio Showcase</span>
          <div style="margin-top:0.9rem;padding:0.65rem;background:#F0FDF4;border-radius:8px;">
            <p style="font-size:0.75rem;color:#14532D;margin:0;"><strong>Typical Setup:</strong> <code style="background:#BBF7D0;padding:1px 5px;border-radius:3px;">streamlit run app.py</code> → Deploy free on Streamlit Community Cloud</p>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Decision matrix
    st.markdown("#### Decision Matrix — When to Use What")
    matrix_data = [
        ("Time to first deployment",      "~2–3 days",         "< 30 minutes"),
        ("Requires frontend knowledge",   "Yes (HTML/React)",  "No"),
        ("Free hosting available",        "Limited",           "✅ Yes (Streamlit Cloud)"),
        ("Suitable for production scale", "✅ Yes",            "⚠️ Limited"),
        ("REST API support",              "✅ Native",         "❌ No"),
        ("Built-in charts & widgets",     "❌ No",             "✅ Yes"),
        ("Custom authentication",         "✅ Yes",            "⚠️ Basic"),
        ("Portfolio showcase",            "⚠️ Needs frontend", "✅ Excellent"),
        ("Mobile app integration",        "✅ Yes (JSON API)", "❌ No"),
        ("Data science prototyping",      "⚠️ Overkill",       "✅ Perfect"),
    ]
    rows_html = "".join([
        f'<tr style="background:{"white" if i%2==0 else "#F8FAFC"};">'
        f'<td style="padding:0.45rem 0.8rem;border-bottom:1px solid #F1F5F9;font-weight:500;">{r[0]}</td>'
        f'<td style="padding:0.45rem 0.8rem;border-bottom:1px solid #F1F5F9;text-align:center;color:#C0392B;font-weight:600;">{r[1]}</td>'
        f'<td style="padding:0.45rem 0.8rem;border-bottom:1px solid #F1F5F9;text-align:center;color:#166534;font-weight:600;">{r[2]}</td>'
        f'</tr>'
        for i, r in enumerate(matrix_data)
    ])
    st.markdown(f"""
    <div class="icard">
    <table class="mtable">
      <thead><tr><th>Criteria</th><th>Flask API</th><th>Streamlit</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # Code tabs
    st.markdown("#### 📄 Production-Ready Code Templates")
    code_tab1, code_tab2, code_tab3, code_tab4 = st.tabs(["Flask API (app.py)", "Streamlit App (app.py)", "requirements.txt", "Dockerfile"])

    with code_tab1:
        st.code('''from flask import Flask, request, jsonify
import joblib, numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model   = joblib.load("models/svm_model.pkl")
scaler  = joblib.load("models/scaler.pkl")
imputer = joblib.load("models/imputer.pkl")

CLASS_NAMES = {1:"Psoriasis", 2:"Seboreic Dermatitis",
               3:"Lichen Planus", 4:"Pityriasis Rosea",
               5:"Chronic Dermatitis", 6:"Pityriasis Rubra Pilaris"}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "accuracy": 0.973})

@app.route("/predict", methods=["POST"])
def predict():
    data     = request.get_json()
    features = [data[f] for f in FEATURE_COLS]
    X        = np.array(features).reshape(1, -1)
    X_imp    = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    pred     = int(model.predict(X_scaled)[0])
    proba    = model.predict_proba(X_scaled)[0]
    return jsonify({
        "predicted_class":   pred,
        "predicted_disease": CLASS_NAMES[pred],
        "confidence":        round(float(max(proba)), 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
''', language="python")

    with code_tab2:
        st.code('''import streamlit as st
import joblib, numpy as np

st.set_page_config(page_title="Skin Disease Predictor",
                   page_icon="🧬", layout="wide")

@st.cache_resource
def load_model():
    return (joblib.load("models/svm_model.pkl"),
            joblib.load("models/scaler.pkl"),
            joblib.load("models/imputer.pkl"))

model, scaler, imputer = load_model()

# Sidebar inputs
with st.sidebar:
    st.header("Patient Features")
    erythema = st.select_slider("Erythema", [0,1,2,3])
    scaling  = st.select_slider("Scaling",  [0,1,2,3])
    age      = st.number_input("Age", 0, 120, 35)
    # ... all 34 features ...

if st.button("Predict", type="primary"):
    X       = np.array([erythema, scaling, age]).reshape(1,-1)
    X_imp   = imputer.transform(X)
    X_sc    = scaler.transform(X_imp)
    pred    = model.predict(X_sc)[0]
    st.success(f"Predicted: {CLASS_NAMES[pred]}")
''', language="python")

    with code_tab3:
        st.code('''# requirements.txt
scikit-learn
numpy
pandas
joblib
streamlit
plotly
flask
flask-cors
gunicorn
''', language="text")

    with code_tab4:
        st.code('''FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir flask flask-cors gunicorn scikit-learn numpy joblib
COPY app/flask_app.py app/flask_app.py
COPY models/ models/
EXPOSE 8000
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "app.flask_app:app"]
''', language="dockerfile")

    # Recommendation box
    st.markdown("""
    <div class="rec-box">
        <div class="rec-icon">🏆</div>
        <div>
            <h3>Recommendation for Your Portfolio</h3>
            <p>Start with <strong>Streamlit</strong> — deploy free on Streamlit Community Cloud in under 10 minutes and share the live link directly with evaluators and recruiters. Then wrap the same model in a <strong>Flask REST API</strong> and containerise it with Docker to demonstrate production-readiness. Having both shows you understand the full deployment spectrum from rapid prototyping to enterprise architecture.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — ARCHITECTURE
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### ML Pipeline Architecture")
    st.markdown("End-to-end flow from raw patient data to deployed prediction service.")

    # Flow diagram
    st.markdown("""
    <div class="icard" style="margin-bottom:1.2rem;">
        <div class="ict">Prediction Pipeline (Inference)</div>
        <div class="flow-wrap">
            <div class="flow-step">
                <div class="fs-icon">📋</div>
                <div class="fs-lbl">Raw Input</div>
                <div class="fs-sub">34 feature values<br>from doctor/form</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="fs-icon">🩹</div>
                <div class="fs-lbl">Imputation</div>
                <div class="fs-sub">SimpleImputer<br>median strategy</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="fs-icon">📏</div>
                <div class="fs-lbl">Scaling</div>
                <div class="fs-sub">StandardScaler<br>fit on train only</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step" style="background:#D4F5F0;border:2px solid #0D7377;">
                <div class="fs-icon">🤖</div>
                <div class="fs-lbl" style="color:#0D7377;">SVM (RBF)</div>
                <div class="fs-sub">C=1.0, gamma=scale<br>97.3% accuracy</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">
                <div class="fs-icon">📊</div>
                <div class="fs-lbl">Probabilities</div>
                <div class="fs-sub">Platt scaling<br>6-class output</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step" style="background:#FFF3CD;">
                <div class="fs-icon">🏥</div>
                <div class="fs-lbl">Prediction</div>
                <div class="fs-sub">Disease class +<br>confidence score</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline cards
    st.markdown("""
    <div class="pipeline-grid">
        <div class="pc">
            <h4>🔧 Model Serialization</h4>
            <p>Save trained artifacts using <code>joblib.dump()</code>. Three files: <code>svm_model.pkl</code>, <code>scaler.pkl</code>, <code>imputer.pkl</code>. Save separately to avoid data leakage.</p>
        </div>
        <div class="pc" style="border-left-color:#F4A823;">
            <h4>⚠️ Data Leakage Prevention</h4>
            <p>Scaler is <code>fit_transform</code> on train only, <code>transform</code> on test and at inference. Never fit on the full dataset — the most common production ML mistake.</p>
        </div>
        <div class="pc" style="border-left-color:#8E44AD;">
            <h4>📦 Versioning Strategy</h4>
            <p>Tag models with version + date: <code>svm_v1_20260317.pkl</code>. Use MLflow or DVC. Always log: accuracy, F1, training date, feature list, scikit-learn version.</p>
        </div>
        <div class="pc" style="border-left-color:#27AE60;">
            <h4>✅ Input Validation</h4>
            <p>At API level, validate: all 34 features present, ordinal features in range <code>[0,3]</code>, Age in <code>[0,120]</code>, family_history in <code>{0,1}</code>.</p>
        </div>
        <div class="pc" style="border-left-color:#E74C3C;">
            <h4>📈 Monitoring in Production</h4>
            <p>Track prediction distribution drift over time. Alert if model predicts one class &gt;60% of the time. Log every prediction with timestamp. Retrain quarterly.</p>
        </div>
        <div class="pc" style="border-left-color:#0B1F3A;">
            <h4>🔒 Clinical Safety</h4>
            <p>Always display confidence score alongside prediction. Add disclaimer: "Supplementary tool — not a replacement for clinical diagnosis." Log all predictions for audit trail compliance.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Recommended stack
    st.markdown("""
    <div class="icard">
        <div class="ict">Recommended Production Stack</div>
        <div class="stack-grid">
            <div class="stack-item">
                <div class="si">🔬</div>
                <div class="sl">Model Layer</div>
                <div class="ss">scikit-learn SVM<br>joblib serialization</div>
            </div>
            <div class="stack-item">
                <div class="si">🌶️</div>
                <div class="sl">API Layer</div>
                <div class="ss">Flask + Gunicorn<br>REST JSON API</div>
            </div>
            <div class="stack-item">
                <div class="si">🐳</div>
                <div class="sl">Container Layer</div>
                <div class="ss">Docker<br>Cloud Run / ECS</div>
            </div>
            <div class="stack-item">
                <div class="si">📊</div>
                <div class="sl">Monitoring</div>
                <div class="ss">MLflow tracking<br>Prometheus metrics</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
