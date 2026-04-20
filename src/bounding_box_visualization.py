import cv2
import numpy as np


class BoundingBoxVisualization:
    def __init__(self, class_names: dict):
        self.class_names = class_names

    def run(self, image: np.ndarray, boxes: list) -> np.ndarray:
        """
        Draw bounding boxes on the image and return the annotated image.
        """
        display_img = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cls_id = int(box[4])
            conf = box[5]
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{self.class_names[cls_id]}: {conf:.2f}"
            cv2.putText(
                display_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1
            )
        return display_img

    def show(self, image: np.ndarray, boxes: list) -> None:
        """
        Draw boxes and display the result using OpenCV.
        """
        annotated = self.run(image, boxes)
        cv2.imshow("Detection Results", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
