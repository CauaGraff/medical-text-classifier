import os
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data.loader import TEXT_COLUMN, TARGET_COLUMN, iter_dataset, load_labels


def profile(path: str, chunksize: int) -> None:
    total = 0
    missing_text = 0
    missing_target = 0
    counts: Counter[int] = Counter()
    text_lengths: list[int] = []

    for chunk in pd.read_csv(path, chunksize=chunksize):
        total += len(chunk)
        missing_text += int(chunk[TEXT_COLUMN].isna().sum())
        missing_target += int(chunk[TARGET_COLUMN].isna().sum())

        cleaned = chunk[[TARGET_COLUMN, TEXT_COLUMN]].dropna()
        counts.update(cleaned[TARGET_COLUMN].astype(int).tolist())
        text_lengths.extend(cleaned[TEXT_COLUMN].astype(str).str.len().tolist())

        print(f"Scanned {total:,} rows...")

    print("\nSummary")
    print(f"Rows: {total:,}")
    print(f"Missing text: {missing_text:,}")
    print(f"Missing target: {missing_target:,}")
    print(f"Class counts: {dict(sorted(counts.items()))}")

    if text_lengths:
        series = pd.Series(text_lengths)
        print(f"Text length mean: {series.mean():.2f}")
        print(f"Text length median: {series.median():.2f}")
        print(f"Text length p95: {series.quantile(0.95):.2f}")


if __name__ == "__main__":
    chunksize = int(os.getenv("CSV_CHUNK_SIZE", "5000"))
    labels_path = os.getenv("LABELS_PATH", "data/raw/medical_tc_labels.csv")
    train_path = os.getenv("TRAIN_PATH", "data/raw/medical_tc_train.csv")
    test_path = os.getenv("TEST_PATH", "data/raw/medical_tc_test.csv")

    print("Labels:")
    print(load_labels(labels_path))

    for name, path in [("TRAIN", train_path), ("TEST", test_path)]:
        if Path(path).exists():
            print(f"\n=== {name} ===")
            profile(path, chunksize)
        else:
            print(f"\n{name} not found: {path}")
