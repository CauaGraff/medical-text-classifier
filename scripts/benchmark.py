import json
import os
import statistics
import time
from pathlib import Path

import joblib
import pandas as pd


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = int(round((len(ordered) - 1) * p))
    return ordered[index]


if __name__ == "__main__":
    model_path = os.getenv("MODEL_PATH", "models/model.joblib")
    test_path = os.getenv("TEST_PATH", "data/raw/medical_tc_test.csv")

    model = joblib.load(model_path)
    sample = pd.read_csv(test_path, nrows=200)["medical_abstract"].dropna().astype(str).tolist()

    timings_ms: list[float] = []
    for text in sample:
        started = time.perf_counter()
        model.predict([text])
        timings_ms.append((time.perf_counter() - started) * 1000)

    results = {
        "requests": len(timings_ms),
        "mean_ms": statistics.mean(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": percentile(timings_ms, 0.95),
        "p99_ms": percentile(timings_ms, 0.99),
    }

    Path("benchmarks").mkdir(exist_ok=True)
    Path("benchmarks/latency.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(results)
