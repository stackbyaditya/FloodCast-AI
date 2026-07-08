from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "floodcast_xgboost.pkl"

print("Loading model from:", MODEL_PATH)

MODEL = joblib.load(MODEL_PATH)


def predict(data):
    prediction = MODEL.predict(data)
    return float(prediction[0])