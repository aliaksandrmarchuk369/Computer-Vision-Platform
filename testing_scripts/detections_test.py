from src.image_processor import ImageProcessor
from src.yolo11 import YOLO11

model_path = "models/yolo11n.onnx"
model = YOLO11(model_path, conf_threshold=0.5)

processor = ImageProcessor()
batch = processor(["images/image_3.jpg"])  # returns numpy array (1,3,640,640)
original_shapes = processor.original_shapes   # list of (height, width)
detections_list = model(batch, original_shapes)

print(detections_list)