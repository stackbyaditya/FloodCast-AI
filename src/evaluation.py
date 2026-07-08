import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def evaluate(y_true, pred):

    return {

        "RMSE":
        np.sqrt(
            mean_squared_error(
                y_true,
                pred
            )
        ),

        "MAE":
        mean_absolute_error(
            y_true,
            pred
        ),

        "R2":
        r2_score(
            y_true,
            pred
        )

    }