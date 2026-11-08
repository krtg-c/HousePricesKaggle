# House Prices — Advanced Regression Techniques

Учебный проект по задаче регрессии с Kaggle. Модель должна предсказать
стоимость дома `SalePrice` по его характеристикам.

## Данные и метрика

Данные находятся в папке `data/`:

- `train.csv` — 1460 домов с известной ценой;
- `test.csv` — 1459 домов для submission;
- `sample_submission.csv` — пример формата результата.

В модели используется 79 признаков. Колонка `Id` нужна только для submission
и не используется при обучении.

Цена имеет длинный правый хвост, поэтому модель обучается предсказывать:

```python
#np.log(SalePrice)
```

Основная метрика — RMSE между логарифмами настоящих и предсказанных цен.
В проекте она называется Log RMSE. Чем меньше значение, тем лучше.

## Установка

Создать и активировать виртуальное окружение:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux или macOS:

```bash
source .venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

При первом запуске TabPFN библиотеке может потребоваться загрузить веса
предобученной модели.

## Основной запуск

```bash
python main.py
```

За один запуск программа:

1. загружает train;
2. логарифмирует `SalePrice`;
3. считает Log RMSE выбранной модели на 10 фолдах;
4. обучает эту же модель на всём train;
5. сохраняет модель в `models/`;
6. создаёт submission в `submissions/`.

Например, для модели `xgboost` создаются:

```text
models/xgboost.joblib
submissions/submission_xgboost.csv
```

## Выбор модели

Модель выбирается в `config.py`:

```python
MODEL_NAME = "xgboost"
```

Доступны варианты:

```python
MODEL_NAME = "linear_regression"
MODEL_NAME = "random_forest"
MODEL_NAME = "xgboost"
MODEL_NAME = "xgboost_huber"
MODEL_NAME = "xgboost_weighted"
MODEL_NAME = "catboost"
MODEL_NAME = "mlp"
MODEL_NAME = "tabpfn"
```

Все параметры моделей, число фолдов и random state также находятся
в `config.py`.

`xgboost_weighted` увеличивает вес домов, у которых `SaleCondition`
не равен `Normal`. Это отдельный вариант обучения одной модели.

## Предобработка

Для sklearn-моделей, XGBoost и MLP:

- числовые пропуски заполняются медианой;
- категориальные пропуски заменяются строкой `Missing`;
- категории кодируются с помощью OneHotEncoder;
- для Linear Regression и MLP числа дополнительно масштабируются.

CatBoost получает категориальные признаки напрямую. TabPFN получает
категории после OrdinalEncoder.

Вся предобработка находится внутри pipeline, поэтому во время CV она
обучается отдельно внутри каждого фолда и не использует validation заранее.

## Простой baseline

Отдельный простой baseline запускается командой:

```bash
python baseline.py
```

Это обычная Linear Regression с той же подготовкой данных и 10-fold CV.

## Результаты

Все модели оценены на одном `KFold` с 10 фолдами, перемешиванием данных
и `random_state=42`.

1. | Модель | Mean CV | Std CV |
2. | Linear Regression | 0.14819 | 0.04572 |
3. | Random Forest | 0.14052 | 0.02441 |
4. | MLP | 0.14199 | 0.02572 |
5. | CatBoost | 0.12155 | 0.02091 |
6. | XGBoost weighted | 0.12073 | 0.02061 |
7. | XGBoost | 0.11865 | 0.02051 |
8. | XGBoost Huber | 0.11856 | 0.01975 |
9. | TabPFN | 0.10886 | 0.01967 |

TabPFN показал лучшую метрику, но на CPU один полный CV занимает несколько
минут. CatBoost с 1600 итерациями также работает долго. По умолчанию выбран
XGBoost, потому что он даёт хороший результат и быстро воспроизводится.

Сохранённый TabPFN занимает около 837 МБ, так как `joblib` сохраняет вместе
с pipeline веса предобученной модели. Остальные модели заметно меньше.

## Структура проекта

- `main.py` — CV, полное обучение и создание submission;
- `config.py` — выбор модели и параметры;
- `train.py` — общие pipeline, CV и сохранение моделей;
- `predict.py` — обратное преобразование цены и создание submission;
- `baseline.py` — Linear Regression baseline;
- `EDA.ipynb` — анализ данных с графиками и выводами;
- `experiments/` — отдельные эксперименты;
- `experiments/mlp.py` — реализация PyTorch-нейросети и её эксперимент;
- `experiments.md` — таблица результатов.
