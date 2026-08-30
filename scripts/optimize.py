from pathlib import Path
import os

import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_classifier_to_onnx(
    model_path: str,
    output_path: str,
) -> None:
    """Export only the trained SGDClassifier to ONNX."""

    print(f"Loading baseline model: {model_path}")

    pipeline = joblib.load(model_path)

    classifier = pipeline.named_steps["classifier"]

    n_features = int(classifier.coef_.shape[1])

    print(f"Classifier: {type(classifier).__name__}")
    print(f"Number of features: {n_features:,}")

    initial_types = [
        (
            "X",
            FloatTensorType(
                [None, n_features]
            ),
        )
    ]

    print("Converting classifier to ONNX...")

    onnx_model = convert_sklearn(
        classifier,
        initial_types=initial_types,
        target_opset={
            "": 18,
            "ai.onnx.ml": 3,
        },
        options={
            id(classifier): {
                "zipmap": False,
            }
        },
    )

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "wb") as file:
        file.write(
            onnx_model.SerializeToString()
        )

    print("Validating ONNX model...")

    model = onnx.load(output_path)

    onnx.checker.check_model(model)

    size_mb = (
        Path(output_path).stat().st_size
        / 1024
        / 1024
    )

    print("ONNX model created successfully.")
    print(f"Saved at: {output_path}")
    print(f"Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    export_classifier_to_onnx(
        model_path=os.getenv(
            "MODEL_PATH",
            "models/model.joblib",
        ),
        output_path=os.getenv(
            "ONNX_MODEL_PATH",
            "models/classifier.onnx",
        ),
    )