import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_model():
    model_path = os.path.join(MODELS_DIR, "yolo11n.pt")
    print(f"Loading model from {model_path} (will download if not present)")
    yolo_model = YOLO(model_path)
    pt_model = yolo_model.model
    pt_model.eval()
    return pt_model

def export_to_onnx(pt_model, onnx_path: str):
    dummy_input = torch.randn(1, 3, 640, 640)
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    print(f"Exporting to {onnx_path} with dynamic batch size...")
    torch.onnx.export(
        pt_model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=11,
        do_constant_folding=True,
        external_data=False,
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX export successful and validated.")

def test_onnx_runs(onnx_path: str):
    test_input = np.random.randn(2, 3, 640, 640).astype(np.float32)
    ort_session = ort.InferenceSession(onnx_path)
    ort_input = {ort_session.get_inputs()[0].name: test_input}
    ort_output = ort_session.run(None, ort_input)[0]
    print(f"ONNX test passed with batch size 2. Output shape: {ort_output.shape}")
    return True

def main():
    pt_model = load_model()
    onnx_path = os.path.join(MODELS_DIR, "yolo11n.onnx")
    export_to_onnx(pt_model, onnx_path)
    test_onnx_runs(onnx_path)

if __name__ == "__main__":
    main()