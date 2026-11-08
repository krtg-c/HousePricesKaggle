from config import MODEL_NAME, MODEL_PATH
from predict import create_submission
from train import (
    create_pipeline,
    evaluate_model,
    load_train_data,
    train_and_save_model
)


def main():
    # 1. Загружаем данные и создаём выбранную модель
    X, y = load_train_data()
    pipeline = create_pipeline(MODEL_NAME, X)

    # 2. Считаем Log RMSE на кросс-валидации
    cv_scores = evaluate_model(pipeline, X, y, MODEL_NAME)

    print(MODEL_NAME.upper())

    for fold, score in enumerate(cv_scores, start=1):
        print(f"Fold {fold}: {score:.5f}")

    print(f"\nMean CV: {cv_scores.mean():.5f}")
    print(f"Std CV: {cv_scores.std():.5f}")

    # 3. Обучаем эту же модель на всех данных
    model_data = train_and_save_model(
        pipeline,
        X,
        y,
        MODEL_NAME
    )

    print(f"\nКоличество домов: {len(X)}")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Модель сохранена: {MODEL_PATH}")

    # 4. Создаём submission этой же моделью
    create_submission(model_data)


if __name__ == "__main__":
    main()
