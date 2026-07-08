from pathlib import Path

# Project Root
ROOT = Path(__file__).resolve().parent.parent

# Data
RAW_DATA = ROOT / "data" / "raw"
PROCESSED_DATA = ROOT / "data" / "processed"

# Models
MODEL_DIR = ROOT / "models"

# Outputs
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"