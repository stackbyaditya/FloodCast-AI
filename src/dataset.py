import pandas as pd

from src.preprocessing import build_preprocessor


def load_training_data():

    master = pd.read_csv("../data/processed/features_dataset.csv")

    TARGET = "Peak Flood Level (m)"

    leakage_columns = [
        "Peak Flood Level (m)",
        "Peak Discharge Q (cumec)",
        "Flood Volume (cumec)",
        "Peak FL Date",
        "Peak Discharge Date",
        "End Date",
        "Event Duration (days)",
        "Num Peak FL",
        "Time to Peak (days)",
        "Recession Time (day)"
    ]

    X = master.drop(columns=leakage_columns)

    y = master[TARGET]

    X["Start Date"] = pd.to_datetime(X["Start Date"])

    date_features = pd.DataFrame({
        "Year": X["Start Date"].dt.year,
        "Month": X["Start Date"].dt.month,
        "DayOfYear": X["Start Date"].dt.dayofyear
    }, index=X.index)

    X = pd.concat(
        [X.drop(columns=["Start Date"]), date_features],
        axis=1
    )

    X.drop(
        columns=["EventID", "GaugeID"],
        inplace=True,
        errors="ignore"
    )

    preprocessor = build_preprocessor(X)

    return X, y, preprocessor