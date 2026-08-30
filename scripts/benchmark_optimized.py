import json
import os
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    ordered = sorted(values)

    if not ordered:
        return 0.0

    index = int(
        round(
            (len(ordered) - 1)
            * percentile_value
        )
    )

    return ordered[index]


def measure_baseline(
    model,
    texts: list[str],
) -> tuple[list[float], list[int]]:
    timings: list[float] = []
    predictions: list[int] = []

    # Warm-up
    for text in texts[:10]:
        model.predict([text])

    for text in texts:
        started = time.perf_counter()

        prediction = model.predict(
            [text]
        )[0]

        elapsed = (
            time.perf_counter() - started
        ) * 1000

        timings.append(elapsed)
        predictions.append(
            int(prediction)
        )

    return timings, predictions


def measure_onnx(
    vectorizer,
    session: ort.InferenceSession,
    texts: list[str],
) -> tuple[list[float], list[int]]:
    timings: list[float] = []
    predictions: list[int] = []

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    # Warm-up
    for text in texts[:10]:
        features = (
            vectorizer
            .transform([text])
            .astype(np.float32)
            .toarray()
        )

        session.run(
            None,
            {
                input_name: features,
            },
        )

    for text in texts:
        started = time.perf_counter()

        features = (
            vectorizer
            .transform([text])
            .astype(np.float32)
            .toarray()
        )

        outputs = session.run(
            None,
            {
                input_name: features,
            },
        )

        prediction = int(
            np.asarray(
                outputs[0]
            ).ravel()[0]
        )

        elapsed = (
            time.perf_counter() - started
        ) * 1000

        timings.append(elapsed)
        predictions.append(prediction)

    return timings, predictions


def summarize(
    timings: list[float],
) -> dict[str, float]:
    return {
        "requests": len(timings),
        "mean_ms": statistics.mean(
            timings
        ),
        "median_ms": statistics.median(
            timings
        ),
        "p95_ms": percentile(
            timings,
            0.95,
        ),
        "p99_ms": percentile(
            timings,
            0.99,
        ),
    }


def improvement(
    baseline: float,
    optimized: float,
) -> float:
    if baseline == 0:
        return 0.0

    return (
        (baseline - optimized)
        / baseline
        * 100
    )


if __name__ == "__main__":
    model_path = os.getenv(
        "MODEL_PATH",
        "models/model.joblib",
    )

    onnx_path = os.getenv(
        "ONNX_MODEL_PATH",
        "models/classifier.onnx",
    )

    test_path = os.getenv(
        "TEST_PATH",
        "data/raw/medical_tc_test.csv",
    )

    print(
        "Loading baseline model..."
    )

    pipeline = joblib.load(
        model_path
    )

    vectorizer = (
        pipeline
        .named_steps["vectorizer"]
    )

    print(
        "Loading ONNX Runtime..."
    )

    session = ort.InferenceSession(
        onnx_path,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    dataframe = pd.read_csv(
        test_path,
        nrows=200,
    )

    texts = (
        dataframe["medical_abstract"]
        .dropna()
        .astype(str)
        .tolist()
    )

    print(
        f"Benchmarking {len(texts)} abstracts..."
    )

    print(
        "\nRunning baseline..."
    )

    baseline_times, baseline_predictions = (
        measure_baseline(
            pipeline,
            texts,
        )
    )

    print(
        "Running ONNX..."
    )

    onnx_times, onnx_predictions = (
        measure_onnx(
            vectorizer,
            session,
            texts,
        )
    )

    baseline_results = summarize(
        baseline_times
    )

    optimized_results = summarize(
        onnx_times
    )

    agreement = (
        np.mean(
            np.array(
                baseline_predictions
            )
            == np.array(
                onnx_predictions
            )
        )
        * 100
    )

    results = {
        "baseline": baseline_results,
        "onnx": optimized_results,
        "prediction_agreement_percent": float(
            agreement
        ),
        "improvement_percent": {
            "mean": improvement(
                baseline_results["mean_ms"],
                optimized_results["mean_ms"],
            ),
            "median": improvement(
                baseline_results[
                    "median_ms"
                ],
                optimized_results[
                    "median_ms"
                ],
            ),
            "p95": improvement(
                baseline_results["p95_ms"],
                optimized_results["p95_ms"],
            ),
            "p99": improvement(
                baseline_results["p99_ms"],
                optimized_results["p99_ms"],
            ),
        },
    }

    Path(
        "benchmarks"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = Path(
        "benchmarks/"
        "onnx_comparison.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n==============================")
    print("       BENCHMARK RESULT")
    print("==============================")

    print("\nBASELINE")
    print(
        json.dumps(
            baseline_results,
            indent=2,
        )
    )

    print("\nONNX")
    print(
        json.dumps(
            optimized_results,
            indent=2,
        )
    )

    print(
        "\nPrediction agreement: "
        f"{agreement:.2f}%"
    )

    print("\nLATENCY IMPROVEMENT")

    for metric, value in (
        results[
            "improvement_percent"
        ].items()
    ):
        print(
            f"{metric}: {value:.2f}%"
        )

    print(
        "\nResults saved at:"
    )

    print(output_path)