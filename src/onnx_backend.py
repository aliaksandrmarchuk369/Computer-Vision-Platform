import onnxruntime as ort
import numpy as np


class ONNXBackend:
    """
    Simple ONNX model wrapper for inference.
    """

    def __init__(self, model_path: str):
        """
        Initialize ONNX Runtime session.

        Args:
            model_path: Path to the .onnx file.
        """
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on the given input.

        Args:
            input_data: NumPy array matching model input shape.

        Returns:
            NumPy array of model outputs.
        """
        ort_input = {self.input_name: input_data}
        outputs = self.session.run(self.output_names, ort_input)

        return outputs[0]
