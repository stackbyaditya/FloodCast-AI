from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor


def get_models():

    return {

        "Linear Regression":
        LinearRegression(),

        "Decision Tree":
        DecisionTreeRegressor(random_state=42),

        "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost":
        XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        ),

        "LightGBM":
        LGBMRegressor(
            random_state=42
        ),

        "CatBoost":
        CatBoostRegressor(
            verbose=0,
            random_state=42
        )

    }