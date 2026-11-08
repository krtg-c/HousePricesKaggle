from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


# "linear_regression", "random_forest", "xgboost",
# "xgboost_huber", "xgboost_weighted", "catboost", "mlp", "tabpfn"
MODEL_NAME = "xgboost"


TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
MODEL_PATH = BASE_DIR / "models" / f"{MODEL_NAME}.joblib"
SUBMISSION_PATH = (
    BASE_DIR / "submissions" / f"submission_{MODEL_NAME}.csv"
)


TARGET = "SalePrice"
ID_COLUMN = "Id"
CV_FOLDS = 10
RANDOM_STATE = 42


MODEL_PARAMS = {
    "linear_regression": {},
    "random_forest": {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "xgboost": {
        "n_estimators": 1000,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": 4
    },
    "xgboost_huber": {
        "n_estimators": 1000,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:pseudohubererror",
        "huber_slope": 0.2,
        "base_score": 12.0,
        "max_delta_step": 1,
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": 4
    },
    "xgboost_weighted": {
        "n_estimators": 1000,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": 4
    },
    "catboost": {
        "iterations": 1600,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 3,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "boost_from_average": True,
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
        "verbose": False
    },
    "mlp": {
        "epochs": 150,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "random_state": RANDOM_STATE
    },
    "tabpfn": {
        "device": "auto",
        "n_estimators": 4,
        "random_state": RANDOM_STATE,
        "show_progress_bar": False
    }
}


SUPPORTED_MODELS = list(MODEL_PARAMS.keys())
