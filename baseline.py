import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CV_FOLDS,
    ID_COLUMN,
    RANDOM_STATE,
    TARGET,
    TRAIN_PATH
)


# 1. Загружаем данные и логарифмируем цену
train = pd.read_csv(TRAIN_PATH)
X = train.drop(columns=[ID_COLUMN, TARGET])
y = np.log(train[TARGET])


# 2. Определяем типы признаков
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()
categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()


# 3. Настраиваем предобработку
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


# 4. Создаём простой baseline
baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])


# 5. Считаем Log RMSE
cv = KFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scores = -cross_val_score(
    baseline,
    X,
    y,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)


print("LINEAR REGRESSION BASELINE")
print(f"Mean CV: {scores.mean():.5f}")
print(f"Std CV: {scores.std():.5f}")
print(f"Min CV: {scores.min():.5f}")
print(f"Max CV: {scores.max():.5f}")
