from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from train import (  # noqa: E402
    create_pipeline,
    evaluate_model,
    load_train_data
)


X, y = load_train_data()
pipeline = create_pipeline("catboost", X)
scores = evaluate_model(pipeline, X, y, "catboost")


print("CATBOOST")

for fold, score in enumerate(scores, start=1):
    print(f"Fold {fold}: {score:.5f}")

print(f"\nMean CV: {scores.mean():.5f}")
print(f"Std CV: {scores.std():.5f}")
print(f"Min CV: {scores.min():.5f}")
print(f"Max CV: {scores.max():.5f}")
