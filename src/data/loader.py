from collections.abc import Iterator
from pathlib import Path

import pandas as pd

TEXT_COLUMN = "medical_abstract"
TARGET_COLUMN = "condition_label"
LABEL_NAME_COLUMN = "condition_name"


def validate_training_schema(dataframe: pd.DataFrame) -> None:
    required = {TARGET_COLUMN, TEXT_COLUMN}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing training/test columns: {sorted(missing)}")


def validate_labels_schema(dataframe: pd.DataFrame) -> None:
    required = {TARGET_COLUMN, LABEL_NAME_COLUMN}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    validate_training_schema(dataframe)
    return dataframe[[TARGET_COLUMN, TEXT_COLUMN]].dropna()


def iter_dataset(path: str | Path, chunksize: int = 5000) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=chunksize):
        validate_training_schema(chunk)
        cleaned = chunk[[TARGET_COLUMN, TEXT_COLUMN]].dropna()
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
        cleaned[TEXT_COLUMN] = cleaned[TEXT_COLUMN].astype(str)
        yield cleaned


def load_labels(path: str | Path) -> dict[int, str]:
    dataframe = pd.read_csv(path)
    validate_labels_schema(dataframe)
    dataframe = dataframe[[TARGET_COLUMN, LABEL_NAME_COLUMN]].dropna()
    return {
        int(row[TARGET_COLUMN]): str(row[LABEL_NAME_COLUMN])
        for _, row in dataframe.iterrows()
    }
