from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# 1. Загружаем обучающие данные
BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE_DIR / "data" / "train.csv"

train = pd.read_csv(TRAIN_PATH)


# 2. Разделяем признаки и целевую переменную
X = train.drop(columns=["Id", "SalePrice"])
y = train["SalePrice"]

# Логарифмируем цены
y_log = np.log(y)


# 3. Определяем типы признаков
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()


# 4. Предобработка числовых признаков
numeric_transformer = SimpleImputer(
    strategy="median"
)


# 5. Предобработка категориальных признаков
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


# 6. Объединяем предобработку
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# 7. Создаём модель
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# 8. Объединяем предобработку и модель
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# 9. Настраиваем кросс-валидацию
cv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


# 10. Запускаем кросс-валидацию
scores = -cross_val_score(
    pipeline,
    X,
    y_log,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)


# 11. Выводим результаты
print("\nRANDOM FOREST")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.5f}")

print(f"\nMean CV: {scores.mean():.5f}")
print(f"Std CV: {scores.std():.5f}")
print(f"Min CV: {scores.min():.5f}")
print(f"Max CV: {scores.max():.5f}")
