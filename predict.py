import joblib
import numpy as np
import pandas as pd

from config import (
    ID_COLUMN,
    MODEL_PATH,
    SUBMISSION_PATH,
    TEST_PATH
)


def create_submission(model_data=None):
    """Создаёт submission с ценами в исходном масштабе."""
    test = pd.read_csv(TEST_PATH)

    if model_data is None:
        model_data = joblib.load(MODEL_PATH)

    pipeline = model_data["pipeline"]
    features = model_data["features"]
    model_name = model_data["model_name"]

    predictions_log = pipeline.predict(test[features])

    assert model_data["target_transform"] == "log"
    predictions = np.exp(predictions_log)

    submission = pd.DataFrame({
        ID_COLUMN: test[ID_COLUMN],
        "SalePrice": predictions
    })

    assert submission.shape == (len(test), 2)
    assert submission[ID_COLUMN].is_unique
    assert np.isfinite(submission["SalePrice"]).all()
    assert (submission["SalePrice"] > 0).all()

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"\n{model_name.upper()} SUBMISSION")
    print(submission.head())
    print(f"\nКоличество домов: {len(submission)}")
    print(f"Минимальная цена: ${predictions.min():,.2f}")
    print(f"Средняя цена: ${predictions.mean():,.2f}")
    print(f"Максимальная цена: ${predictions.max():,.2f}")
    print(f"\nФайл сохранён: {SUBMISSION_PATH}")

    return submission


if __name__ == "__main__":
    create_submission()
