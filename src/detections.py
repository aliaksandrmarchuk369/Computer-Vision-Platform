from dataclasses import dataclass
import numpy as np


@dataclass
class Detections:
    """
    Container for detection results.
    boxes: List of numpy arrays, each of shape (6) with columns:
       x1, y1, x2, y2, class_id, confidence_score
    """

    boxes: list[np.ndarray]
