import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data.loader import iter_dataset
from src.training.train import train_model


def validate_data() -> None:
    train_path = os.getenv("TRAIN_PATH", "data/raw/medical_tc_train.csv")
    chunksize = int(os.getenv("CSV_CHUNK_SIZE", "5000"))

    first = next(iter_dataset(train_path, chunksize=chunksize), None)
    if first is None or first.empty:
        raise ValueError("Training dataset is empty or invalid.")


def train() -> None:
    train_model(
        train_path=os.getenv("TRAIN_PATH", "data/raw/medical_tc_train.csv"),
        output_path=os.getenv("MODEL_PATH", "models/model.joblib"),
        mode=os.getenv("MODEL_MODE", "streaming"),
        chunksize=int(os.getenv("CSV_CHUNK_SIZE", "5000")),
    )


with DAG(
    dag_id="medical_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "medical"],
) as dag:
    validate = PythonOperator(task_id="validate_data", python_callable=validate_data)
    training = PythonOperator(task_id="train_and_save_model", python_callable=train)
    validate >> training
