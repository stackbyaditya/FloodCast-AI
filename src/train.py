from sklearn.pipeline import Pipeline


def train_pipeline(
    model,
    preprocessor,
    X_train,
    y_train,
):
    """
    Build and train a complete sklearn pipeline.
    """

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    return pipe