import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data.loader import TEXT_COLUMN, TARGET_COLUMN, iter_dataset


def evaluate_model(
    model_path: str,
    test_path: str,
    output_path: str,
    chunksize: int = 5000,
) -> dict[str, float]:
    model = joblib.load(model_path)

    all_targets: list[int] = []
    all_predictions: list[int] = []

    total_rows = 0
    for chunk in iter_dataset(test_path, chunksize=chunksize):
        predictions = model.predict(chunk[TEXT_COLUMN])
        all_targets.extend(chunk[TARGET_COLUMN].astype(int).tolist())
        all_predictions.extend(int(x) for x in predictions)
        total_rows += len(chunk)
        print(f"Evaluated {total_rows:,} rows...")

    if not all_targets:
        raise ValueError("No valid test rows were found.")

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    results = {
        "accuracy": float(accuracy_score(all_targets, all_predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "evaluated_rows": float(len(all_targets)),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
