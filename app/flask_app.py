"""
Skin Disease Prediction — Flask REST API
========================================
SVM (RBF Kernel) | UCI Dermatology Dataset | 97.3% Accuracy

Run locally:
    pip install -r requirements.txt
    python app/flask_app.py

Production:
    gunicorn -w 4 -b 0.0.0.0:8000 app.flask_app:app

API Endpoints:
    GET  /health        → Model status
    POST /predict       → Predict disease from 34 features
    GET  /classes       → List all disease classes
    GET  /features      → List all required features
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load model artifacts ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model   = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))

with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
    METADATA = json.load(f)

CLASS_NAMES = {int(k): v for k, v in METADATA["classes"].items()}
FEATURE_COLS = METADATA["features"]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — returns model status and accuracy."""
    return jsonify({
        "status": "ok",
        "model": METADATA["model"],
        "test_accuracy": METADATA["test_accuracy"],
        "f1_macro": METADATA["f1_macro"],
        "training_samples": METADATA["training_samples"]
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    """Returns all 6 disease class names."""
    return jsonify({
        "classes": CLASS_NAMES,
        "total": len(CLASS_NAMES)
    })


@app.route("/features", methods=["GET"])
def get_features():
    """Returns all required feature names with their valid ranges."""
    features_info = []
    for feat in FEATURE_COLS:
        if feat == "Age":
            features_info.append({"name": feat, "type": "numeric", "range": [0, 120]})
        elif feat == "family_history":
            features_info.append({"name": feat, "type": "binary", "range": [0, 1]})
        else:
            features_info.append({"name": feat, "type": "ordinal", "range": [0, 3]})
    return jsonify({"features": features_info, "total": len(features_info)})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict skin disease from patient features.

    Request body (JSON):
    {
        "erythema": 2,
        "scaling": 2,
        "definite_borders": 1,
        ...all 34 features...
        "Age": 45
    }

    Response:
    {
        "predicted_class": 1,
        "predicted_disease": "Psoriasis",
        "confidence": 0.92,
        "all_probabilities": { "Psoriasis": 0.92, ... }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        # Validate and extract features
        features = []
        missing = []
        out_of_range = []

        for feat in FEATURE_COLS:
            val = data.get(feat)
            if val is None:
                missing.append(feat)
                features.append(np.nan)
                continue

            val = float(val)

            # Range validation
            if feat == "Age" and not (0 <= val <= 120):
                out_of_range.append(f"{feat} must be 0–120")
            elif feat == "family_history" and val not in [0, 1]:
                out_of_range.append(f"{feat} must be 0 or 1")
            elif feat not in ["Age", "family_history"] and not (0 <= val <= 3):
                out_of_range.append(f"{feat} must be 0–3")

            features.append(val)

        if out_of_range:
            return jsonify({"error": "Validation failed", "details": out_of_range}), 400

        # Preprocess
        X = np.array(features).reshape(1, -1)
        X_imp    = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)

        # Predict
        pred_class = int(model.predict(X_scaled)[0])
        pred_proba = model.predict_proba(X_scaled)[0]
        classes    = model.classes_

        all_proba = {
            CLASS_NAMES[int(cls)]: round(float(p), 4)
            for cls, p in zip(classes, pred_proba)
        }

        return jsonify({
            "predicted_class":   pred_class,
            "predicted_disease": CLASS_NAMES[pred_class],
            "confidence":        round(float(max(pred_proba)), 4),
            "all_probabilities": all_proba,
            "missing_features":  missing,
            "note": "Missing features were imputed with training medians."
                    if missing else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧬 Skin Disease Prediction API")
    print(f"   Model: {METADATA['model']}")
    print(f"   Accuracy: {METADATA['test_accuracy']:.1%}")
    print("   Running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
