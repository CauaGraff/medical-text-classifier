import os
from pathlib import Path

for path in [
    os.getenv("TRAIN_PATH", "data/raw/medical_tc_train.csv"),
    os.getenv("TEST_PATH", "data/raw/medical_tc_test.csv"),
]:
    print(f"{path}: {'OK' if Path(path).exists() else 'MISSING'}")
