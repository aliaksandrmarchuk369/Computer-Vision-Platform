import cv2
import numpy as np


class BoundingBoxVisualization:
    def __init__(self, class_names: dict):
        self.class_names = class_names

    def run(self, image: np.ndarray, boxes: list) -> np.ndarray:
        display_img = image.copy()
        img_w = display_img.shape[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cls_id = int(box[4])
            conf = box[5]
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{self.class_names[cls_id]}: {conf:.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Default: above the box
            label_x = x1
            label_y = y1 - 5

            # If label would be cut off at top, place inside box near top edge
            if label_y - text_h < 0:
                label_y = y1 + text_h + 5

            # Horizontal adjustments
            if label_x < 0:
                label_x = 5
            if label_x + text_w > img_w:
                label_x = img_w - text_w - 5

            cv2.putText(
                display_img, label, (label_x, label_y), font, font_scale, (0, 0, 0), thickness
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
