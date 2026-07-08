from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer


def build_preprocessor(X):

    numeric = X.select_dtypes(
        include=["int64","float64"]
    ).columns

    categorical = X.select_dtypes(
        include="object"
    ).columns

    numeric_pipeline = Pipeline([

        ("imputer",
         SimpleImputer(strategy="median")),

        ("scaler",
         StandardScaler())

    ])

    categorical_pipeline = Pipeline([

        ("imputer",
         SimpleImputer(strategy="most_frequent")),

        ("encoder",
         OneHotEncoder(handle_unknown="ignore"))

    ])

    preprocessor = ColumnTransformer([

        ("num",
         numeric_pipeline,
         numeric),

        ("cat",
         categorical_pipeline,
         categorical)

    ])

    return preprocessor