"""
train_model.py — Retrain and save the SVM model
================================================
Run this script if you want to retrain the model from scratch:

    python train_model.py

Outputs:
    models/svm_model.pkl
    models/scaler.pkl
    models/imputer.pkl
    models/model_metadata.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    1: "Psoriasis", 2: "Seboreic Dermatitis", 3: "Lichen Planus",
    4: "Pityriasis Rosea", 5: "Chronic Dermatitis", 6: "Pityriasis Rubra Pilaris"
}

DATA_PATH   = os.path.join("data", "dataset_35_dermatology__1_.csv")
MODEL_DIR   = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_clean(path):
    df = pd.read_csv(path)
    # Fix Age column: '?' → NaN → median imputed
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    print(f"✅ Loaded: {df.shape[0]} records, {df.shape[1]} columns")
    print(f"   Missing values after cleaning: {df.isnull().sum().sum()}")
    return df


def train(df):
    feature_cols = [c for c in df.columns if c not in ["class"]]
    X = df[feature_cols].values
    y = df["class"].values

    print(f"\n📊 Class distribution:")
    for cls, name in CLASS_NAMES.items():
        print(f"   Class {cls} ({name}): {(y == cls).sum()}")

    # Preprocessing
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X)
    scaler  = StandardScaler()
    X_scl   = scaler.fit_transform(X_imp)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scl, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✂️  Split: {len(X_train)} train / {len(X_test)} test")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(kernel="rbf", probability=True, random_state=42, C=1.0, gamma="scale")
    cv_scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"\n📈 CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final training
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="macro")
    prec = precision_score(y_test, y_pred, average="macro")
    rec  = recall_score(y_test, y_pred, average="macro")

    print(f"\n🏆 Test Results:")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1 Macro  : {f1:.4f}")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=list(CLASS_NAMES.values()))}")

    return svm, scaler, imputer, feature_cols, {
        "model": "SVM_RBF",
        "test_accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "features": feature_cols,
        "classes": {str(k): v for k, v in CLASS_NAMES.items()},
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "scaler": "StandardScaler",
        "imputer": "SimpleImputer(median)",
        "hyperparameters": {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
    }


def save_artifacts(model, scaler, imputer, metadata):
    joblib.dump(model,   os.path.join(MODEL_DIR, "svm_model.pkl"))
    joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n💾 Artifacts saved to {MODEL_DIR}/")
    print(f"   ✓ svm_model.pkl")
    print(f"   ✓ scaler.pkl")
    print(f"   ✓ imputer.pkl")
    print(f"   ✓ model_metadata.json")


if __name__ == "__main__":
    print("🧬 Skin Disease Prediction — Model Training")
    print("=" * 50)
    df = load_and_clean(DATA_PATH)
    model, scaler, imputer, feature_cols, metadata = train(df)
    save_artifacts(model, scaler, imputer, metadata)
    print("\n✅ Training complete!")
