from pathlib import Path
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

def export_to_onnx(model_path: str, output_path: str) -> None:
    model = joblib.load(model_path)
    initial_type = [("text", StringTensorType([None, 1]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(onnx_model.SerializeToString())
