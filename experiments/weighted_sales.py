from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from typing import cast


# 1. Загружаем данные
BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE_DIR / "data" / "train.csv"

train = pd.read_csv(TRAIN_PATH)


# 2. Выделяем признаки и целевую переменную
X = train.drop(columns=["Id", "SalePrice"])
y = np.log(train["SalePrice"])


# 3. Определяем типы признаков
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()


# 4. Предобработка числовых данных
numeric_transformer = SimpleImputer(
    strategy="median"
)


# 5. Предобработка категорий
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Missing"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])


# 6. Объединяем предобработку
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# 7. Создаём модель XGBoost
xgboost_model = XGBRegressor(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    random_state=42,
    n_jobs=4
)


# 8. Объединяем обработку и модель
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgboost_model)
])


# 9. Кросс-валидация
cv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


# 10. Обучение и оценка
scores = -cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)

# SAMPLE WEIGHTS

weighted_scores = []
weighted_predictions = np.empty(len(X))

for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y), start=1
):
    # Разделяем текущий фолд
    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # Создаём веса только для обучающих домов
    weights = np.where(
        train.iloc[train_idx]["SaleCondition"] == "Normal",
        1.0,
        2.0
    )

    # Создаём новую модель для каждого фолда
    weighted_model = cast(Pipeline, clone(pipeline))

    # Обучаем с учётом весов
    weighted_model.fit(
        X_train,
        y_train,
        model__sample_weight=weights
    )

    # Предсказываем логарифмы цен
    predictions = weighted_model.predict(X_valid)

    weighted_predictions[valid_idx] = predictions

    # Оцениваем на всех проверочных домах
    rmse = np.sqrt(
        mean_squared_error(y_valid, predictions)
    )

    weighted_scores.append(rmse)

    print(f"Fold {fold}: {rmse:.5f}")


weighted_scores = np.array(weighted_scores)

print("\nWEIGHTED XGBOOST")
print(f"Baseline CV: {scores.mean():.5f}")
print(f"Weighted CV: {weighted_scores.mean():.5f}")
print(f"Std: {weighted_scores.std():.5f}")

print(
    f"Изменение RMSE: "
    f"{weighted_scores.mean() - scores.mean():+.5f}"
)



# Обычные out-of-fold предсказания baseline
baseline_predictions = cross_val_predict(
    pipeline,
    X,
    y,
    cv=cv
)

y_values = y.to_numpy()


groups = {
    "Normal": train["SaleCondition"]
        .eq("Normal")
        .to_numpy(),

    "Non-normal": train["SaleCondition"]
        .ne("Normal")
        .to_numpy()
}


print("\nERRORS BY SALE CONDITION")

for group_name, mask in groups.items():

    baseline_error = np.sqrt(np.mean(
        (y_values[mask] - baseline_predictions[mask]) ** 2
    ))

    weighted_error = np.sqrt(np.mean(
        (y_values[mask] - weighted_predictions[mask]) ** 2
    ))

    print(f"\n{group_name}")
    print(f"Baseline RMSE: {baseline_error:.5f}")
    print(f"Weighted RMSE: {weighted_error:.5f}")
