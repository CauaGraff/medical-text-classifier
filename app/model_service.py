import os
from pathlib import Path
from typing import Any

import joblib

from src.data.loader import load_labels


class ModelService:
    def __init__(
        self,
        model_path: str | None = None,
        labels_path: str | None = None,
    ) -> None:
        self.model_path = Path(model_path or os.getenv("MODEL_PATH", "models/model.joblib"))
        self.labels_path = Path(
            labels_path or os.getenv("LABELS_PATH", "data/raw/medical_tc_labels.csv")
        )
        self.model: Any = None
        self.labels: dict[int, str] = {}

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run scripts/train.py first."
            )
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found at {self.labels_path}.")
        self.model = joblib.load(self.model_path)
        self.labels = load_labels(self.labels_path)

    def predict(self, text: str) -> tuple[int, str, float]:
        if self.model is None:
            self.load()

        prediction = int(self.model.predict([text])[0])

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([text])[0]
            confidence = float(max(probabilities))
        else:
            confidence = 0.0

        return prediction, self.labels.get(prediction, "unknown"), confidence
