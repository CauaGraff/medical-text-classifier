import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort

from src.data.loader import load_labels


class ModelService:
    def __init__(
        self,
        model_path: str | None = None,
        onnx_model_path: str | None = None,
        labels_path: str | None = None,
    ) -> None:
        self.model_path = Path(
            model_path
            or os.getenv(
                "MODEL_PATH",
                "models/model.joblib",
            )
        )

        self.onnx_model_path = Path(
            onnx_model_path
            or os.getenv(
                "ONNX_MODEL_PATH",
                "models/classifier.onnx",
            )
        )

        self.labels_path = Path(
            labels_path
            or os.getenv(
                "LABELS_PATH",
                "data/raw/medical_tc_labels.csv",
            )
        )

        self.pipeline: Any = None
        self.vectorizer: Any = None
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None

        self.labels: dict[int, str] = {}

    def load(self) -> None:
        """Load vectorizer, ONNX model and labels."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Baseline model not found at {self.model_path}."
            )

        if not self.onnx_model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {self.onnx_model_path}."
            )

        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found at {self.labels_path}."
            )

        self.pipeline = joblib.load(
            self.model_path
        )

        self.vectorizer = (
            self.pipeline
            .named_steps["vectorizer"]
        )

        self.session = ort.InferenceSession(
            str(self.onnx_model_path),
            providers=[
                "CPUExecutionProvider",
            ],
        )

        self.input_name = (
            self.session
            .get_inputs()[0]
            .name
        )

        self.labels = load_labels(
            self.labels_path
        )

    def predict(
        self,
        text: str,
    ) -> tuple[int, str, float]:
        """Predict using HashingVectorizer + ONNX Runtime."""

        if (
            self.vectorizer is None
            or self.session is None
            or self.input_name is None
        ):
            self.load()

        features = (
            self.vectorizer
            .transform([text])
            .astype(np.float32)
            .toarray()
        )

        outputs = self.session.run(
            None,
            {
                self.input_name: features,
            },
        )

        prediction = int(
            np.asarray(
                outputs[0]
            ).ravel()[0]
        )

        confidence = self._extract_confidence(
            outputs
        )

        class_name = self.labels.get(
            prediction,
            "unknown",
        )

        return (
            prediction,
            class_name,
            confidence,
        )

    @staticmethod
    def _extract_confidence(
        outputs: list[Any],
    ) -> float:
        """Extract the highest probability returned by ONNX."""

        if len(outputs) < 2:
            return 0.0

        probabilities = np.asarray(
            outputs[1]
        )

        if probabilities.size == 0:
            return 0.0

        return float(
            np.max(probabilities)
        )