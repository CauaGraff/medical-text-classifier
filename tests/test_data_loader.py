import pandas as pd
import pytest

from src.data.loader import validate_labels_schema, validate_training_schema


def test_training_schema() -> None:
    dataframe = pd.DataFrame({
        "condition_label": [1],
        "medical_abstract": ["example"],
    })
    validate_training_schema(dataframe)


def test_labels_schema() -> None:
    dataframe = pd.DataFrame({
        "condition_label": [1],
        "condition_name": ["neoplasms"],
    })
    validate_labels_schema(dataframe)


def test_invalid_training_schema() -> None:
    with pytest.raises(ValueError):
        validate_training_schema(pd.DataFrame({"wrong": [1]}))
