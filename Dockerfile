# ── Dockerfile — Flask API for Skin Disease Classifier ───────────────────────
# Build:   docker build -t skin-disease-api .
# Run:     docker run -p 8000:8000 skin-disease-api
# Deploy:  Google Cloud Run / AWS ECS / Azure Container Apps

FROM python:3.11-slim

# Metadata
LABEL maintainer="Niranjan Shinde"
LABEL description="Skin Disease Prediction API – SVM (RBF Kernel)"
LABEL version="1.0"

WORKDIR /app

# Install dependencies (separate layer for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir flask flask-cors gunicorn scikit-learn numpy joblib

# Copy source code and model artifacts
COPY app/flask_app.py app/flask_app.py
COPY models/ models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with Gunicorn (production WSGI server, 4 workers)
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", \
     "--timeout", "60", "--access-logfile", "-", "app.flask_app:app"]
