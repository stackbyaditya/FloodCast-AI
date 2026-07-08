from datetime import datetime

import pandas as pd

from fastapi import FastAPI
from fastapi import HTTPException

from app.schemas import PredictionRequest
from app.predictor import predict

app = FastAPI(
    title="Flood AI API",
    version="1.0.0",
    description="""
Flood Prediction API

Predicts Peak Flood Level using the trained XGBoost model.
""",
)

# ---------------------------------------------------
# Home
# ---------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Flood AI API Running 🚀",
        "model": "XGBoost",
        "version": "1.0.0"
    }


# ---------------------------------------------------
# Health Check
# ---------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "time": datetime.now().isoformat()
    }


# ---------------------------------------------------
# Model Information
# ---------------------------------------------------

@app.get("/model-info")
def model_info():

    return {

        "Model": "XGBoost",

        "Target": "Peak Flood Level (m)",

        "Version": "1.0.0",

        "Author": "Gaganpreet Singh",

        "Framework": "Scikit-Learn Pipeline + XGBoost"

    }


# ---------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------

@app.post("/predict")
def predict_flood(request: PredictionRequest):

    try:

        df = pd.DataFrame([request.features])

        prediction = predict(df)

        return {

            "status": "success",

            "prediction": {

                "Peak Flood Level (m)": round(prediction, 2)

            },

            "model": "XGBoost",

            "version": "1.0.0",

            "timestamp": datetime.now().isoformat()

        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )