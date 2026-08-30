import os

from src.training.train import train_model

if __name__ == "__main__":
    train_path = os.getenv("TRAIN_PATH", "data/raw/medical_tc_train.csv")
    model_path = os.getenv("MODEL_PATH", "models/model.joblib")
    mode = os.getenv("MODEL_MODE", "streaming")
    chunksize = int(os.getenv("CSV_CHUNK_SIZE", "5000"))

    print(f"Training file: {train_path}")
    print(f"Mode: {mode}")
    print(f"Chunk size: {chunksize}")

    train_model(train_path, model_path, mode=mode, chunksize=chunksize)
    print(f"Model saved at: {model_path}")
