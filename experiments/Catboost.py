from pathlib import Path
import numpy as np
import pandas as pd
from catboost import Pool, cv as catboost_cv
from sklearn.model_selection import KFold

# 1. Загружаем данные
BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE_DIR / "data" / "train.csv"

train = pd.read_csv(TRAIN_PATH)


# 2. Выделяем признаки и целевую переменную
X = train.drop(columns=["Id", "SalePrice"]).copy()
y = np.log(train["SalePrice"])


# 3. Находим категориальные признаки
categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()


# 4. Обрабатываем категориальные пропуски
X[categorical_features] = (
    X[categorical_features]
    .fillna("Missing")
    .astype(str)
)


# 5. Создаём Pool для CatBoost
train_pool = Pool(
    data=X,
    label=y,
    cat_features=categorical_features
)


# 6. Настройки модели
catboost_params = {
    "iterations": 1600,
    "depth": 6,
    "learning_rate": 0.04,
    "l2_leaf_reg": 3,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "boost_from_average": True,
    "random_seed": 42,
    "allow_writing_files": False
}


# 7. Создаём те же 10 фолдов
cv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

folds = list(cv.split(X, y))


# 8. Запускаем кросс-валидацию
cv_results = catboost_cv(
    pool=train_pool,
    params=catboost_params,
    folds=folds,
    early_stopping_rounds=150,
    verbose=False
)


# 9. Получаем итоговую ошибку
catboost_mean = cv_results["test-RMSE-mean"].iloc[-1]
catboost_std = cv_results["test-RMSE-std"].iloc[-1]

print("\n=== EXPERIMENT 1: CATBOOST ===")
print(f"Mean CV: {catboost_mean:.5f}")
print(f"Std CV: {catboost_std:.5f}")


# 10. Сравниваем с random forest

print("\n=== COMPARISON ===")
print(f"Random Forest: 0.14052")
print(f"CatBoost: {catboost_mean:.5f}")

difference = catboost_mean - 0.14052

print(f"Изменение RMSE: {difference:+.5f}")


# === АНАЛИЗ ОБУЧЕНИЯ CATBOOST ===

print("\n=== TRAIN VS VALIDATION ===")

for n in [50, 100, 200, 400, 600, 800]:

    train_rmse = cv_results["train-RMSE-mean"].iloc[n - 1]
    valid_rmse = cv_results["test-RMSE-mean"].iloc[n - 1]

    print(
        f"{n} деревьев: "
        f"Train RMSE = {train_rmse:.5f}, "
        f"Validation RMSE = {valid_rmse:.5f}"
    )

# Ищем итерацию с минимальной ошибкой
best_idx = cv_results["test-RMSE-mean"].idxmin()

best_iteration = int(
    cv_results.loc[best_idx, "iterations"]
) + 1

best_rmse = cv_results.loc[
    best_idx, "test-RMSE-mean"
]

print("\n=== BEST ITERATION ===")
print(f"Лучшая итерация: {best_iteration}")
print(f"Минимальный CV RMSE: {best_rmse:.5f}")

best_idx = cv_results["test-RMSE-mean"].idxmin()

best_iteration = int(
    cv_results.loc[best_idx, "iterations"]
) + 1

best_rmse = cv_results.loc[
    best_idx, "test-RMSE-mean"
]

best_std = cv_results.loc[
    best_idx, "test-RMSE-std"
]

print("\n=== CATBOOST: EARLY STOPPING ===")
print(f"Выполнено итераций: {len(cv_results)}")
print(f"Лучшая итерация: {best_iteration}")
print(f"Best CV RMSE: {best_rmse:.5f}")
print(f"Std CV: {best_std:.5f}")
