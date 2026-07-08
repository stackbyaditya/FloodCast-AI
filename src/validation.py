from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def cross_validate_model(
    pipeline,
    X,
    y,
    cv=5
):

    kfold = KFold(
        n_splits=cv,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=kfold,
        scoring="r2",
        n_jobs=-1
    )

    return {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std()
    }