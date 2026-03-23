# 🧬 Skin Disease Prediction — ML Classifier

> **Erythemato-Squamous Disease Classification using SVM (RBF Kernel)**  
> UCI Dermatology Dataset · 97.3% Test Accuracy · 6 Disease Classes

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

This project builds a complete machine learning pipeline to classify **6 erythemato-squamous skin diseases** using clinical and histopathological patient features from the UCI Dermatology dataset.

All 6 diseases share the clinical features of erythema and scaling, making differential diagnosis challenging even for experienced dermatologists. This model provides a data-driven second opinion to assist clinical decision-making.

### 6 Disease Classes
| Class | Disease |
|-------|---------|
| 1 | Psoriasis |
| 2 | Seboreic Dermatitis |
| 3 | Lichen Planus |
| 4 | Pityriasis Rosea |
| 5 | Chronic Dermatitis |
| 6 | Pityriasis Rubra Pilaris |

---

## 🏆 Results Summary

| Model | CV Accuracy | Test Accuracy | F1 Macro |
|-------|------------|---------------|----------|
| **SVM (RBF) ← Best** | **97.60% ± 0.009** | **97.30%** | **96.97%** |
| Logistic Regression | 97.60% | 95.95% | 95.74% |
| Random Forest | 97.25% | 95.95% | 95.45% |
| Extra Trees | 97.94% | 95.95% | 95.45% |
| Gradient Boosting | 96.57% | 93.24% | 91.25% |
| Decision Tree | 95.55% | 93.24% | 88.78% |
| K-Nearest Neighbors | 96.91% | 90.54% | 90.34% |
| Naive Bayes | 89.72% | 86.49% | 82.24% |
| AdaBoost | 55.06% | 64.86% | 45.91% |

---

## 📂 Project Structure

```
skin-disease-prediction/
│
├── 📓 notebooks/
│   └── Skin_Disease_Prediction.ipynb   ← Complete EDA + ML pipeline
│
├── 🌐 app/
│   ├── flask_app.py                    ← REST API (production)
│   ├── streamlit_app.py                ← Interactive web app (demo)
│   └── SkinDiseasePredictor.html       ← Standalone browser demo
│
├── 🤖 models/
│   ├── svm_model.pkl                   ← Trained SVM model
│   ├── scaler.pkl                      ← StandardScaler
│   ├── imputer.pkl                     ← SimpleImputer
│   └── model_metadata.json             ← Model info & metrics
│
├── 📊 data/
│   └── dataset_35_dermatology__1_.csv  ← UCI Dermatology dataset
│
├── 📋 requirements.txt
├── 🐳 Dockerfile
└── 📖 README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/iamniranjan7/skin-disease-prediction.git
cd skin-disease-prediction
pip install -r requirements.txt
```

### 2. Run the Streamlit App (Recommended for demo)

```bash
streamlit run app/streamlit_app.py
```
Opens at `http://localhost:8501` — adjust sliders and click **Run Prediction**.

### 3. Run the Flask REST API

```bash
python app/flask_app.py
```
API runs at `http://localhost:5000`

```bash
# Test the API
curl -X GET http://localhost:5000/health

curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "erythema": 2, "scaling": 2, "definite_borders": 1,
       "itching": 1, "koebner_phenomenon": 0, "polygonal_papules": 0,
       "follicular_papules": 0, "oral_mucosal_involvement": 0,
       "knee_and_elbow_involvement": 2, "scalp_involvement": 2,
       "family_history": 1, "melanin_incontinence": 0,
       "eosinophils_in_the_infiltrate": 0, "PNL_infiltrate": 2,
       "fibrosis_of_the_papillary_dermis": 0, "exocytosis": 1,
       "acanthosis": 2, "hyperkeratosis": 0, "parakeratosis": 2,
       "clubbing_of_the_rete_ridges": 2, "elongation_of_the_rete_ridges": 2,
       "thinning_of_the_suprapapillary_epidermis": 2, "spongiform_pustule": 0,
       "munro_microabcess": 1, "focal_hypergranulosis": 0,
       "disappearance_of_the_granular_layer": 0,
       "vacuolisation_and_damage_of_basal_layer": 0, "spongiosis": 0,
       "saw-tooth_appearance_of_retes": 0, "follicular_horn_plug": 0,
       "perifollicular_parakeratosis": 0, "inflammatory_monoluclear_inflitrate": 1,
       "band-like_infiltrate": 0, "Age": 45
     }'
```

**Response:**
```json
{
  "predicted_class": 1,
  "predicted_disease": "Psoriasis",
  "confidence": 0.9412,
  "all_probabilities": {
    "Psoriasis": 0.9412,
    "Seboreic Dermatitis": 0.0231,
    ...
  }
}
```

### 4. Run with Docker

```bash
docker build -t skin-disease-api .
docker run -p 8000:8000 skin-disease-api
```

### 5. Open the Standalone Browser Demo

Simply open `app/SkinDiseasePredictor.html` in any browser — no server required.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Dermatology) |
| Records | 366 patients |
| Features | 34 (11 clinical + 22 histopathological + 1 demographic) |
| Target | 6 disease classes |
| Missing Values | 8 (`Age` = `?`) → Median imputed |
| Class Range | Ordinal 0–3 (0=absent, 3=most severe) |

---

## 🔬 Methodology

### Data Preprocessing
- **Missing values:** 8 `Age` records had `?` → replaced with median (35 years) using `SimpleImputer`
- **Scaling:** `StandardScaler` applied only to SVM, KNN, Logistic Regression (tree models are scale-invariant)
- **Splitting:** 80/20 stratified train-test split to preserve class proportions

### Model Selection
- 9 models evaluated: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Extra Trees, SVM, KNN, Naive Bayes
- **Evaluation metric:** F1-Macro (equally weights all 6 classes regardless of imbalance)
- **Cross-validation:** 5-fold Stratified K-Fold
- **Tuning:** GridSearchCV on Random Forest

### Why SVM Wins
- Highest test accuracy (97.3%) and lowest variance (CV std = 0.009)
- RBF kernel captures non-linear boundaries between disease classes that share erythema/scaling features
- Effective in high-dimensional space with limited samples (366 records, 34 features)

### Top Diagnostic Features (Random Forest Importance)
1. Clubbing of rete ridges (0.0990)
2. Fibrosis of papillary dermis (0.0766)
3. Thinning of suprapapillary epidermis (0.0702)
4. Koebner phenomenon (0.0639)
5. Spongiosis (0.0596)

> 8 of the top 10 features are **histopathological** — confirming that biopsy-level analysis is essential for accurate differential diagnosis.

---

## 🩺 Clinical Recommendations

| Priority | Recommendation |
|----------|---------------|
| 🔴 HIGH | Prioritise early biopsy for all erythema + scaling cases |
| 🔴 HIGH | Score ALL features on 0–3 scale (never binary) |
| 🟠 MEDIUM | Look for band-like infiltrate → near-perfect Lichen Planus marker (Δ=+1.84) |
| 🟠 MEDIUM | Assess follicular horn plug + perifollicular parakeratosis → exclusive to Pityriasis Rubra |
| 🟠 MEDIUM | Measure PNL infiltrate + scalp involvement → top Psoriasis differentiators |
| 🟢 LOW | Record family history systematically → Psoriasis shows strong hereditary component |
| 🟢 LOW | Integrate ML model as clinical second opinion |

---

## 🐳 Deployment Options

| | Streamlit | Flask API |
|--|-----------|-----------|
| **Setup time** | < 30 min | 2–3 days |
| **Free hosting** | ✅ Streamlit Cloud | ⚠️ Limited |
| **REST API** | ❌ | ✅ |
| **Best for** | Demo, portfolio | Production, integration |

### Deploy Streamlit (Free, 5 minutes)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `app/streamlit_app.py` → Deploy

### Deploy Flask on Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/skin-disease-api
gcloud run deploy skin-disease-api \
    --image gcr.io/PROJECT_ID/skin-disease-api \
    --platform managed --region us-central1 --allow-unauthenticated
```

---

## 📁 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model status + accuracy |
| `/predict` | POST | Predict disease from 34 features |
| `/classes` | GET | List all 6 disease classes |
| `/features` | GET | List all features with valid ranges |

---

## 🛠️ Tech Stack

- **ML:** scikit-learn, NumPy, pandas
- **API:** Flask, Flask-CORS, Gunicorn
- **App:** Streamlit, Plotly
- **Containerisation:** Docker
- **Notebook:** Jupyter, Matplotlib, Seaborn

---

## 👤 Author

**Niranjan Shinde**  
Aspiring Data Scientist | Mumbai, India

- 🐙 GitHub: [github.com/iamniranjan7](https://github.com/iamniranjan7)
- 💼 LinkedIn: [linkedin.com/in/niranjan-shinde7](https://linkedin.com/in/niranjan-shinde7)
- 📊 Kaggle: [kaggle.com/niranjan0705](https://kaggle.com/niranjan0705)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- UCI Machine Learning Repository for the Dermatology dataset
- Original dataset collectors and dermatology researchers
- scikit-learn community for the excellent ML library
