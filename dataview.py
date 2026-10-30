from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Пути к файлам
BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"

# 2. Загрузка данных
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# 3. Размеры выборок
print("РАЗМЕРЫ ДАННЫХ")
print("Train:", train.shape)
print("Test:", test.shape)

# 4. Первые строки
print("\nПЕРВЫЕ 5 СТРОК")
print(train.head().to_string())

# 5. Информация о колонках
print("\nИНФОРМАЦИЯ О ДАННЫХ")
train.info()

# 6. Типы признаков
print("\nТИПЫ ДАННЫХ")
print(train.dtypes.value_counts())

# 7. Пропуски в train
print("\nПРОПУСКИ В TRAIN")

missing_train = (
    train.isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing_train.head(20))

# 8. Пропуски в test
print("\nПРОПУСКИ В TEST")

missing_test = (
    test.isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing_test.head(20))

# 9. Статистика целевой переменной
print("\nСТОИМОСТЬ ДОМОВ")
print(train["SalePrice"].describe())

# 10. Проверка target
print("\nПРОВЕРКА TARGET")
print("SalePrice в train:", "SalePrice" in train.columns)
print("SalePrice в test:", "SalePrice" in test.columns)


# 11. Распределение стоимости домов

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# График 1. Обычное распределение цен
sns.histplot(
    data=train,
    x="SalePrice",
    kde=True,
    ax=axes[0, 0]
)
axes[0, 0].set_title("Распределение цен домов")
axes[0, 0].set_xlabel("Цена, доллары")
axes[0, 0].set_ylabel("Количество домов")

# График 2. Распределение логарифма цены
sns.histplot(
    x=np.log1p(train["SalePrice"]),
    kde=True,
    ax=axes[0, 1]
)
axes[0, 1].set_title("Распределение log(1 + SalePrice)")
axes[0, 1].set_xlabel("log(1 + цена)")
axes[0, 1].set_ylabel("Количество домов")

# График 3. Цена в зависимости от общего качества
sns.boxplot(
    data=train,
    x="OverallQual",
    y="SalePrice",
    ax=axes[1, 0]
)
axes[1, 0].set_title("Цена в зависимости от качества дома")
axes[1, 0].set_xlabel("Общая оценка качества")
axes[1, 0].set_ylabel("Цена, доллары")

# График 4. Цена в зависимости от жилой площади
sns.scatterplot(
    data=train,
    x="GrLivArea",
    y="SalePrice",
    alpha=0.7,
    ax=axes[1, 1]
)
axes[1, 1].set_title("Цена в зависимости от жилой площади")
axes[1, 1].set_xlabel("Жилая площадь, кв. футы")
axes[1, 1].set_ylabel("Цена, доллары")

plt.tight_layout()
plt.show()


# 12. Корреляция признаков с ценой

numeric_data = train.select_dtypes(include="number")

numeric_data = numeric_data.drop(columns=["Id"])

correlations = numeric_data.corr()["SalePrice"]

top_features = (
    correlations.drop("SalePrice")
    .abs()
    .sort_values(ascending=False)
    .head(15)
    .index
)

print("\nТОП-15 ЧИСЛОВЫХ ПРИЗНАКОВ ПО СИЛЕ КОРРЕЛЯЦИИ С ЦЕНОЙ")

for feature in top_features:
    print(f"{feature}: {correlations[feature]:.3f}")


# 13. Тепловая карта корреляций
heatmap_features = list(top_features) + ["SalePrice"]

plt.figure(figsize=(12, 9))

sns.heatmap(
    numeric_data[heatmap_features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Корреляция цены и числовых признаков")
plt.tight_layout()
plt.show()
