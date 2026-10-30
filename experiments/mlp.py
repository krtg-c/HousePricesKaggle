from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Загрузка данных
BASE_DIR = Path(__file__).resolve().parents[1]

train = pd.read_csv(BASE_DIR / "data" / "train.csv")

X = train.drop(columns=["Id", "SalePrice"])
y = np.log(train["SalePrice"].to_numpy())


# 2. Типы признаков
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()


# 3. Настройки обучения
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001

cv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

scores = []


# 4. Кросс-валидация
for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y), start=1
):

    # 4.1. Разделяем данные
    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y[train_idx]
    y_valid = y[valid_idx]


    # 4.2. Подготовка признаков
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]), categorical_features)
    ])

    X_train_processed = preprocessor.fit_transform(
        X_train
    ).astype(np.float32)

    X_valid_processed = preprocessor.transform(
        X_valid
    ).astype(np.float32)


    # 4.3. Нормализуем целевую переменную
    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train_scaled = (
        (y_train - y_mean) / y_std
    ).astype(np.float32)


    # 4.4. Преобразуем данные в тензоры
    X_train_tensor = torch.tensor(X_train_processed)

    y_train_tensor = torch.tensor(
        y_train_scaled
    ).reshape(-1, 1)

    X_valid_tensor = torch.tensor(X_valid_processed)


    # 4.5. Создаём DataLoader
    dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    generator = torch.Generator().manual_seed(42 + fold)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )


    # 4.6. Создаём новую нейросеть
    torch.manual_seed(42 + fold)

    input_size = X_train_processed.shape[1]

    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(64, 32),
        nn.ReLU(),

        nn.Linear(32, 1)
    )


    # 4.7. Функция потерь
    criterion = nn.MSELoss()

    # Оптимизатор
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.001
    )


    # 4.8. Обучение
    for epoch in range(EPOCHS):
        model.train()

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(predictions, y_batch)

            loss.backward()

            optimizer.step()


    # 4.9. Проверка качества
    model.eval()

    with torch.no_grad():
        predictions_scaled = (
            model(X_valid_tensor)
            .numpy()
            .ravel()
        )

    # Возвращаемся к логарифмам цен
    predictions_log = (
        predictions_scaled * y_std + y_mean
    )

    # Вычисляем Log RMSE
    rmse = np.sqrt(
        np.mean((y_valid - predictions_log) ** 2)
    )

    scores.append(rmse)

    print(f"Fold {fold}: {rmse:.5f}")


# 5. Итоги кросс-валидации
scores = np.array(scores)

print("\nMLP")

print(f"Mean CV: {scores.mean():.5f}")
print(f"Std CV: {scores.std():.5f}")
print(f"Min CV: {scores.min():.5f}")
print(f"Max CV: {scores.max():.5f}")

print("\n=== COMPARISON ===")
print("Random Forest: 0.14052")
print("CatBoost:      0.11991")
print("XGBoost:       0.11865")
print(f"MLP: {scores.mean():.5f}")
