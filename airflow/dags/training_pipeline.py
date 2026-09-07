import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data.loader import iter_dataset
from src.training.train import train_model


def validate_data() -> None:
    """Validate that the training dataset exists and contains valid rows."""

    train_path = os.getenv(
        "TRAIN_PATH",
        "/opt/airflow/project/data/raw/medical_tc_train.csv",
    )

    chunksize = int(
        os.getenv(
            "CSV_CHUNK_SIZE",
            "5000",
        )
    )

    if not Path(train_path).exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    first_chunk = next(
        iter_dataset(
            train_path,
            chunksize=chunksize,
        ),
        None,
    )

    if first_chunk is None or first_chunk.empty:
        raise ValueError(
            "Training dataset is empty or invalid."
        )

    print(
        f"Dataset validated successfully. "
        f"First chunk: {len(first_chunk)} rows."
    )


def train_and_save() -> None:
    """Train the streaming model and save it."""

    train_path = os.getenv(
        "TRAIN_PATH",
        "/opt/airflow/project/data/raw/medical_tc_train.csv",
    )

    model_path = os.getenv(
        "MODEL_PATH",
        "/opt/airflow/project/models/model.joblib",
    )

    mode = os.getenv(
        "MODEL_MODE",
        "streaming",
    )

    chunksize = int(
        os.getenv(
            "CSV_CHUNK_SIZE",
            "5000",
        )
    )

    train_model(
        train_path=train_path,
        output_path=model_path,
        mode=mode,
        chunksize=chunksize,
    )

    print(
        f"Model trained and saved at: {model_path}"
    )


with DAG(
    dag_id="medical_training_pipeline",
    description="Medical Abstracts training pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=[
        "machine-learning",
        "medical",
        "training",
    ],
) as dag:

    validate_data_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    train_model_task = PythonOperator(
        task_id="train_and_save_model",
        python_callable=train_and_save,
    )

    validate_data_task >> train_model_task