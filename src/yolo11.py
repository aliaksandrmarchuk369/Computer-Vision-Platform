import numpy as np
import cv2
from src.onnx_backend import ONNXBackend
from src.detections import Detections
import json


class YOLO11:
    def __init__(
        self,
        model_path: str,
        class_names_path: str = "src/yolo11_class_names.json",
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        filter_classes: list = None,
    ):
        self.backend = ONNXBackend(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.filter_classes = filter_classes if filter_classes else []

        with open(class_names_path, "r") as f:
            raw_dict = json.load(f)
        # Convert string keys to int keys
        self.class_names = {int(k): v for k, v in raw_dict.items()}

    def __call__(self, input_data: np.ndarray, original_shapes: list[tuple]) -> list[Detections]:
        """
        Args:
            input_data: (N, 3, 640, 640) batch of preprocessed images.
            original_shapes: List of (height, width) for each image in the batch.
        Returns:
            List of Detections, one per image.
        """
        raw_output = self.backend(input_data)  # (N, 84, 8400)
        results = []
        for i in range(raw_output.shape[0]):
            single_output = raw_output[i : i + 1]  # (1,84,8400)
            boxes_list = self._decode_yolo_outputs(single_output)
            boxes_list = self.postprocess(boxes_list)
            # Rescale boxes to original image dimensions
            h_orig, w_orig = original_shapes[i]
            boxes_list = self._rescale_boxes(boxes_list, w_orig, h_orig)
            results.append(Detections(boxes=boxes_list))
        return results

    def _decode_yolo_outputs(self, raw_output: np.ndarray) -> list:
        predictions = raw_output[0]  # (84, 8400)
        predictions = predictions.T  # (8400, 84)
        cx = predictions[:, 0]
        cy = predictions[:, 1]
        w = predictions[:, 2]
        h = predictions[:, 3]
        class_scores = predictions[:, 4:]  # (8400, 80)
        max_conf = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        detections = [
            np.array([x1[i], y1[i], x2[i], y2[i], class_ids[i], max_conf[i]])
            for i in range(len(x1))
        ]
        return detections

    def postprocess(self, detections: list) -> list:
        detections = self._confidence_threshold(detections)
        detections = self._class_filter(detections)
        detections = self._nms(detections)
        return detections

    def _confidence_threshold(self, detections: list) -> list:
        return [d for d in detections if d[5] >= self.conf_threshold]

    def _class_filter(self, detections: list) -> list:
        if not self.filter_classes:
            return detections
        return [d for d in detections if int(d[4]) in self.filter_classes]

    def _nms(self, detections: list) -> list:
        if not detections:
            return []
        by_class = {}
        for d in detections:
            by_class.setdefault(int(d[4]), []).append(d)
        kept = []
        for cls, boxes in by_class.items():
            bboxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
            scores = [b[5] for b in boxes]
            indices = cv2.dnn.NMSBoxes(bboxes, scores, 0.0, self.nms_threshold)
            if indices is not None and len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0]
                elif isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                kept.extend([boxes[i] for i in indices])
        return kept

    def _rescale_boxes(self, detections: list, orig_width: int, orig_height: int) -> list:
        """
        Scale bounding boxes from model input size (640,640) to original image size.
        """
        scale_x = orig_width / 640.0
        scale_y = orig_height / 640.0
        rescaled = []
        for d in detections:
            x1 = d[0] * scale_x
            y1 = d[1] * scale_y
            x2 = d[2] * scale_x
            y2 = d[3] * scale_y
            rescaled.append(np.array([x1, y1, x2, y2, d[4], d[5]]))
        return rescaled
