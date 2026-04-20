import cv2
import numpy as np


class BoundingBoxVisualization:
    def __init__(self, class_names: dict):
        self.class_names = class_names  # dict mapping int class_id -> str name

    def __call__(self, image: np.ndarray, boxes: list = None):
        display_img = image.copy()
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cls_id = int(box[4])
                conf = box[5]
                # Green bounding box
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Red text
                label = f"{self.class_names[cls_id]}: {conf:.2f}"
                cv2.putText(
                    display_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1
                )
        cv2.imshow("Detection Results", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
