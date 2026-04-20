import cv2
from src.image_processor import ImageProcessor
from src.yolo11 import YOLO11
from src.bounding_box_visualization import BoundingBoxVisualization

# Load model
model = YOLO11("models/yolo11n.onnx", conf_threshold=0.3)

# List of image paths
image_paths = ["images/image_1.jpg", "images/image_2.jpg", "images/image_3.jpg", "images/image_4.jpg"]
#image_paths = ["images/image_3.jpg"]

# Load and preprocess all images at once
processor = ImageProcessor()
batch = processor(image_paths)               # shape (N, 3, 640, 640)
original_shapes = processor.original_shapes  # list of (height, width) for each image

# Run inference (pass both batch and original shapes for rescaling)
detections_list = model(batch, original_shapes)   # list of Detections (one per image)

# Visualize each result
for i, (img_path, detections) in enumerate(zip(image_paths, detections_list)):
    original_image = cv2.imread(img_path)
    visualizer = BoundingBoxVisualization(class_names=model.class_names)
    print(f"Showing detections for {img_path}")
    visualizer(original_image, detections.boxes)
    cv2.waitKey(0)

cv2.destroyAllWindows()