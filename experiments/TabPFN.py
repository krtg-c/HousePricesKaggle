from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor

BASE_DIR = Path(__file__).resolve().parents[1]

train = pd.read_csv(BASE_DIR / "data" / "train.csv")

X = train.drop(columns=["Id", "SalePrice"]).copy()
y = np.log(train["SalePrice"].to_numpy())

# Отмечаем категориальные признаки
for col in X.select_dtypes(
    include=["object", "string"]
).columns:
    X[col] = X[col].astype("category")


cv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

scores = []

for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y), start=1
):
    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y[train_idx]
    y_valid = y[valid_idx]

    model = TabPFNRegressor(
        device="auto",
        n_estimators=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    rmse = np.sqrt(
        np.mean((y_valid - predictions) ** 2)
    )

    scores.append(rmse)

    print(f"Fold {fold}: {rmse:.5f}")


scores = np.array(scores)

print("\n=== TABPFN RESULTS ===")
print(f"Mean CV: {scores.mean():.5f}")
print(f"Std CV: {scores.std():.5f}")
print(f"Min CV: {scores.min():.5f}")
print(f"Max CV: {scores.max():.5f}")
