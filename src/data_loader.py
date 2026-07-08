import pandas as pd

from src.config import RAW_DATA


def load_datasets():

    catchment = pd.read_csv(
        RAW_DATA / "catchment_characteristics_indofloods.csv"
    )

    events = pd.read_csv(
        RAW_DATA / "floodevents_indofloods.csv"
    )

    precip = pd.read_csv(
        RAW_DATA / "precipitation_variables_indofloods.csv"
    )

    return catchment, events, precip