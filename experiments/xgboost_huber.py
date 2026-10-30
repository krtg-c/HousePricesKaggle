from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


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
    objective="reg:pseudohubererror",
    huber_slope=0.2,
    base_score=12.0,
    max_delta_step=1,
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


# 11. Вывод результатов
print("\nXGBOOST pseudohubererror")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.5f}")

print(f"\nMean CV: {scores.mean():.5f}")
print(f"Std CV: {scores.std():.5f}")
print(f"Min CV: {scores.min():.5f}")
print(f"Max CV: {scores.max():.5f}")



