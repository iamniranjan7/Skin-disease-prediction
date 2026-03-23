"""Auto-trains and saves model if pkl files don't exist."""
import os, joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def ensure_model_exists():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'svm_model.pkl')
    
    if os.path.exists(model_path):
        return  # Already trained
    
    print("🔄 Training model for first time...")
    
    # Find dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_35_dermatology__1_.csv')
    
    df = pd.read_csv(data_path)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    feature_cols = [c for c in df.columns if c != 'class']
    X = df[feature_cols].values
    y = df['class'].values
    
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_scaled, y)
    
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
    joblib.dump(model,   model_path)
    joblib.dump(scaler,  model_path.replace('svm_model', 'scaler'))
    joblib.dump(imputer, model_path.replace('svm_model', 'imputer'))
    
    print("✅ Model trained and saved!")