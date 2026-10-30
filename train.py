from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# 1. Пути к файлам
BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_1000.joblib"


# 2. Загружаем данные
train = pd.read_csv(TRAIN_PATH)


# 3. Разделяем признаки и цену
X = train.drop(columns=["Id", "SalePrice"])
y = np.log(train["SalePrice"])


# 4. Определяем типы признаков
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()


# 5. Предобработка
numeric_transformer = SimpleImputer(
    strategy="median"
)

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

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# 6. XGBoost с проверенными параметрами
model = XGBRegressor(
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


# 7. Создаём Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# 8. Обучаем на всех 1460 домах
pipeline.fit(X, y)


# 9. Сохраняем модель и список признаков
MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

model_data = {
    "pipeline": pipeline,
    "features": X.columns.tolist(),
    "target_transform": "log"
}

joblib.dump(model_data, MODEL_PATH)


print("XGBOOST")
print(f"Количество домов: {len(X)}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Модель сохранена: {MODEL_PATH}")
