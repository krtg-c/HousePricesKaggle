from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# 1. Пути к файлам
BASE_DIR = Path(__file__).resolve().parent

TEST_PATH = BASE_DIR / "data" / "test.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_1000.joblib"
SUBMISSION_PATH = BASE_DIR / "submissions" / "submission.csv"


# 2. Загружаем тестовые данные
test = pd.read_csv(TEST_PATH)


# 3. Загружаем обученную модель
model_data = joblib.load(MODEL_PATH)

pipeline = model_data["pipeline"]
features = model_data["features"]
target_transform = model_data["target_transform"]


# 4. Подготавливаем признаки
X_test = test[features]


# 5. Получаем предсказания
predictions_log = pipeline.predict(X_test)


# 6. Преобразуем логарифмы обратно в цены
assert target_transform == "log"

predictions = np.exp(predictions_log)


# 7. Формируем таблицу для Kaggle
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": predictions
})


# 8. Проверяем корректность данных
assert submission.shape == (1459, 2)
assert submission["Id"].is_unique
assert np.isfinite(submission["SalePrice"]).all()
assert (submission["SalePrice"] > 0).all()


# 9. Сохраняем CSV
SUBMISSION_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

submission.to_csv(
    SUBMISSION_PATH,
    index=False
)


# 10. Выводим результаты
print("\nXGBOOST SUBMISSION")

print(submission.head())

print(f"\nКоличество домов: {len(submission)}")
print(f"Минимальная цена: ${predictions.min():,.2f}")
print(f"Средняя цена: ${predictions.mean():,.2f}")
print(f"Максимальная цена: ${predictions.max():,.2f}")

print(f"\nФайл сохранён: {SUBMISSION_PATH}")
