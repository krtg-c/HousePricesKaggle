import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)
from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor

from config import (
    CV_FOLDS,
    ID_COLUMN,
    MODEL_NAME,
    MODEL_PARAMS,
    MODEL_PATH,
    RANDOM_STATE,
    SUPPORTED_MODELS,
    TARGET,
    TRAIN_PATH
)
from experiments.mlp import TorchMLPRegressor


def load_train_data():
    """Загружает данные и логарифмирует цену."""
    train = pd.read_csv(TRAIN_PATH)
    X = train.drop(columns=[ID_COLUMN, TARGET])
    y = np.log(train[TARGET])
    return X, y


def get_feature_types(X):
    """Возвращает списки числовых и категориальных колонок."""
    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns.tolist()
    return numeric_features, categorical_features


def create_preprocessor(X, scale_numeric=False):
    """Создаёт предобработку для sklearn-моделей."""
    numeric_features, categorical_features = get_feature_types(X)

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])


def create_linear_regression_pipeline(X):
    preprocessor = create_preprocessor(
        X,
        scale_numeric=True
    )
    model = LinearRegression(**MODEL_PARAMS["linear_regression"])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def create_random_forest_pipeline(X):
    preprocessor = create_preprocessor(X)
    model = RandomForestRegressor(
        **MODEL_PARAMS["random_forest"]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def create_xgboost_pipeline(X, model_name="xgboost"):
    preprocessor = create_preprocessor(X)
    model = XGBRegressor(**MODEL_PARAMS[model_name])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def prepare_catboost_data(data):
    """Заполняет пропуски в категориальных колонках CatBoost."""
    data = data.copy()
    categorical_features = data.select_dtypes(
        exclude="number"
    ).columns.tolist()
    data[categorical_features] = (
        data[categorical_features]
        .fillna("Missing")
        .astype(str)
    )
    return data


def create_catboost_pipeline(X):
    _, categorical_features = get_feature_types(X)
    preprocessor = FunctionTransformer(
        prepare_catboost_data,
        validate=False
    )
    model = CatBoostRegressor(
        cat_features=categorical_features,
        **MODEL_PARAMS["catboost"]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def create_mlp_pipeline(X):
    preprocessor = create_preprocessor(X, scale_numeric=True)
    model = TorchMLPRegressor(**MODEL_PARAMS["mlp"])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def create_tabpfn_pipeline(X):
    numeric_features, categorical_features = get_feature_types(X)

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )
        ),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ])

    first_categorical_index = len(numeric_features)
    categorical_indices = list(range(
        first_categorical_index,
        first_categorical_index + len(categorical_features)
    ))

    model = TabPFNRegressor(
        categorical_features_indices=categorical_indices,
        **MODEL_PARAMS["tabpfn"]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def create_pipeline(model_name, X):
    """Выбирает модель по значению из config.py."""
    if model_name == "linear_regression":
        return create_linear_regression_pipeline(X)
    if model_name == "random_forest":
        return create_random_forest_pipeline(X)
    if model_name in [
        "xgboost",
        "xgboost_huber",
        "xgboost_weighted"
    ]:
        return create_xgboost_pipeline(X, model_name)
    if model_name == "catboost":
        return create_catboost_pipeline(X)
    if model_name == "mlp":
        return create_mlp_pipeline(X)
    if model_name == "tabpfn":
        return create_tabpfn_pipeline(X)

    raise ValueError(
        f"Неизвестная модель. Доступные модели: {SUPPORTED_MODELS}"
    )


def get_sample_weights(X):
    """Увеличивает вес домов с нестандартным условием продажи."""
    return np.where(X["SaleCondition"] == "Normal", 1.0, 2.0)


def evaluate_model(pipeline, X, y, model_name=MODEL_NAME):
    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    if model_name in ["catboost", "xgboost_weighted"]:
        scores = []

        for train_index, valid_index in cv.split(X, y):
            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]

            fold_pipeline = create_pipeline(model_name, X_train)

            if model_name == "xgboost_weighted":
                weights = get_sample_weights(X_train)
                fold_pipeline.fit(
                    X_train,
                    y_train,
                    model__sample_weight=weights
                )
            else:
                fold_pipeline.fit(X_train, y_train)

            predictions = fold_pipeline.predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, predictions))
            scores.append(rmse)

        return np.array(scores)

    return -cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )


def train_and_save_model(pipeline, X, y, model_name=MODEL_NAME):
    """Обучает выбранную модель на всех данных и сохраняет её."""
    if model_name == "xgboost_weighted":
        weights = get_sample_weights(X)
        pipeline.fit(X, y, model__sample_weight=weights)
    else:
        pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "pipeline": pipeline,
        "features": X.columns.tolist(),
        "target_transform": "log",
        "model_name": model_name
    }

    joblib.dump(model_data, MODEL_PATH)
    return model_data


if __name__ == "__main__":
    X, y = load_train_data()
    pipeline = create_pipeline(MODEL_NAME, X)
    train_and_save_model(pipeline, X, y, MODEL_NAME)

    print(MODEL_NAME.upper())
    print(f"Количество домов: {len(X)}")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Модель сохранена: {MODEL_PATH}")
