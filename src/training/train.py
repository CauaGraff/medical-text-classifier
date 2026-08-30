from pathlib import Path

import joblib
import numpy as np

from src.data.loader import TEXT_COLUMN, TARGET_COLUMN, iter_dataset
from src.preprocessing.pipeline import (
    build_standard_pipeline,
    build_streaming_pipeline,
)

CLASSES = np.array([1, 2, 3, 4, 5], dtype=int)


def train_standard(
    train_path: str,
    output_path: str,
) -> None:
    """Train the standard model using the complete dataset."""
    from src.data.loader import load_dataset

    data = load_dataset(train_path)

    model = build_standard_pipeline()

    model.fit(
        data[TEXT_COLUMN],
        data[TARGET_COLUMN],
    )

    save_model(model, output_path)


def train_streaming(
    train_path: str,
    output_path: str,
    chunksize: int = 5000,
) -> None:
    """Train the model incrementally without loading the full CSV."""

    pipeline = build_streaming_pipeline()

    vectorizer = pipeline.named_steps["vectorizer"]
    classifier = pipeline.named_steps["classifier"]

    first_chunk = True
    total_rows = 0

    for chunk in iter_dataset(
        train_path,
        chunksize=chunksize,
    ):
        texts = chunk[TEXT_COLUMN].tolist()

        targets = chunk[TARGET_COLUMN].to_numpy(
            dtype=int
        )

        features = vectorizer.transform(texts)

        if first_chunk:
            classifier.partial_fit(
                features,
                targets,
                classes=CLASSES,
            )

            first_chunk = False

        else:
            classifier.partial_fit(
                features,
                targets,
            )

        total_rows += len(chunk)

        print(
            f"Processed {total_rows:,} training rows..."
        )

    if first_chunk:
        raise ValueError(
            "No valid training rows were found."
        )

    save_model(
        pipeline,
        output_path,
    )


def save_model(
    model: object,
    output_path: str,
) -> None:
    """Persist the trained model."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        path,
    )


def train_model(
    train_path: str,
    output_path: str,
    mode: str = "streaming",
    chunksize: int = 5000,
) -> None:
    """Select and execute the training strategy."""

    if mode == "standard":
        train_standard(
            train_path,
            output_path,
        )
        return

    if mode == "streaming":
        train_streaming(
            train_path,
            output_path,
            chunksize,
        )
        return

    raise ValueError(
        "MODEL_MODE must be 'standard' or 'streaming'."
    )