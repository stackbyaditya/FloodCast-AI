import joblib
from pathlib import Path


def save_model(model, path):
    """
    Save a trained model to disk.
    """

    path = Path(path)

    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    print(f"✅ Model saved to: {path}")