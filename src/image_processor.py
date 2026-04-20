import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, target_size: tuple = (640, 640)):
        self.target_size = target_size
        self.images = None
        self.original_shapes = None   # list of (height, width) for each image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # Resize
        img = cv2.resize(image, self.target_size)
        # Color conversion
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalization
        img = img.astype(np.float32) / 255.0
        # Transposition: HWC -> CHW
        img = np.transpose(img, (2, 0, 1))   # (3, H, W) – no batch dimension
        return img

    def __call__(self, image_paths: str | list[str]) -> np.ndarray:
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        preprocessed = []
        shapes = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {path}")
            h, w = img.shape[:2]
            shapes.append((h, w))
            preprocessed.append(self.preprocess(img))

        # Stack along new axis 0 to create batch dimension
        self.images = np.stack(preprocessed, axis=0)   # (N, 3, H, W)
        self.original_shapes = shapes
        return self.images