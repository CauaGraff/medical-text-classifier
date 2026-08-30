import os

from src.evaluation.evaluate import evaluate_model

if __name__ == "__main__":
    model_path = os.getenv("MODEL_PATH", "models/model.joblib")
    test_path = os.getenv("TEST_PATH", "data/raw/medical_tc_test.csv")
    chunksize = int(os.getenv("CSV_CHUNK_SIZE", "5000"))

    results = evaluate_model(
        model_path=model_path,
        test_path=test_path,
        output_path="benchmarks/evaluation.json",
        chunksize=chunksize,
    )
    print(results)
