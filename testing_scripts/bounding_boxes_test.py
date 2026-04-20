import cv2

from src.bounding_box_visualization import BoundingBoxVisualization
from src.image_processor import ImageProcessor
from src.yolo11 import YOLO11

# Load model
model = YOLO11()

# List of image paths
image_paths = [
    "images/image_1.jpg",
    "images/image_2.jpg",
    "images/image_3.jpg",
    "images/image_4.jpg",
]

# Load and preprocess all images at once
processor = ImageProcessor()
batch = processor(image_paths)
original_shapes = processor.original_shapes

# Run inference
detections_list = model(batch, original_shapes)

# Visualize each result
for img_path, detections in zip(image_paths, detections_list):
    original_image = cv2.imread(img_path)
    visualizer = BoundingBoxVisualization(class_names=model.class_names)
    print(f"Showing detections for {img_path}")
    visualizer.show(original_image, detections.boxes)  # show handles waitKey and destroy
