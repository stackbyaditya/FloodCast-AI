from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
import optuna
from xgboost import XGBRegressor


def optimize_xgboost(X, y, preprocessor, cv=5, n_trials=40):

    kfold = KFold(
        n_splits=cv,
        shuffle=True,
        random_state=42
    )

    def objective(trial):

        model = XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            learning_rate=trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True
            ),
            max_depth=trial.suggest_int(
                "max_depth",
                3,
                10
            ),
            subsample=trial.suggest_float(
                "subsample",
                0.6,
                1.0
            ),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0
            ),
            min_child_weight=trial.suggest_int(
                "min_child_weight",
                1,
                10
            ),
            random_state=42,
            n_jobs=1
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=kfold,
            scoring="r2",
            n_jobs=-1
        )

        mean_score = float(scores.mean())

        print("=" * 60)
        print("Trial:", trial.number)
        print("Scores:", scores)
        print("Mean :", mean_score)
        print("=" * 60)

        return mean_score

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=n_trials
    )

    return study