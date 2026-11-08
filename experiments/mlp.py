from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from config import (  # noqa: E402
    CV_FOLDS,
    ID_COLUMN,
    MODEL_PARAMS,
    RANDOM_STATE,
    TARGET,
    TRAIN_PATH
)


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    """Небольшая нейросеть-регрессор с интерфейсом sklearn."""

    def __init__(
        self,
        epochs=150,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=0.001,
        random_state=42
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y):
        """Обучает сеть предсказывать нормализованный log(SalePrice)."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std())
        y_scaled = (y - self.y_mean_) / self.y_std_

        torch.manual_seed(self.random_state)

        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y_scaled).reshape(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        generator = torch.Generator().manual_seed(self.random_state)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator
        )

        input_size = X.shape[1]

        self.model_ = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        for _ in range(self.epochs):
            self.model_.train()

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                predictions = self.model_(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Возвращает предсказания в масштабе log(SalePrice)."""
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X)

        self.model_.eval()

        with torch.no_grad():
            predictions_scaled = (
                self.model_(X_tensor).numpy().ravel()
            )

        return (
            predictions_scaled * self.y_std_ + self.y_mean_
        )


def main():
    """Запускает отдельный CV-эксперимент с MLP."""
    train = pd.read_csv(TRAIN_PATH)
    X = train.drop(columns=[ID_COLUMN, TARGET])
    y = np.log(train[TARGET])

    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
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
            ]),
            categorical_features
        )
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            TorchMLPRegressor(**MODEL_PARAMS["mlp"])
        )
    ])

    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = -cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )

    print("MLP")

    for fold, score in enumerate(scores, start=1):
        print(f"Fold {fold}: {score:.5f}")

    print(f"\nMean CV: {scores.mean():.5f}")
    print(f"Std CV: {scores.std():.5f}")
    print(f"Min CV: {scores.min():.5f}")
    print(f"Max CV: {scores.max():.5f}")


if __name__ == "__main__":
    main()
