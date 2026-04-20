import base64
import os
import tempfile
from typing import List, Optional

import cv2
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader
from starlette.templating import _TemplateResponse

from src.bounding_box_visualization import BoundingBoxVisualization
from src.image_processor import ImageProcessor
from src.yolo11 import YOLO11

app = FastAPI()

# Jinja2 setup (unchanged)
env = Environment(loader=FileSystemLoader("templates"), autoescape=True, cache_size=0)


def template_response(name: str, context: dict):
    template = env.get_template(name)
    return _TemplateResponse(template, context)


@app.get("/")
async def home(request: Request):
    return template_response("index.html", {"request": request})


@app.post("/detect")
async def detect(
    files: List[UploadFile] = File(...),
    conf_threshold: float = Query(0.5, ge=0.0, le=1.0),
    nms_threshold: float = Query(0.5, ge=0.0, le=1.0),
    filter_classes: Optional[str] = Query(None, description="Comma-separated class IDs"),
):
    # Parse filter_classes
    filter_list = None
    if filter_classes:
        try:
            filter_list = [int(c.strip()) for c in filter_classes.split(",")]
        except ValueError:
            filter_list = []  # or raise HTTPException

    temp_paths = []
    for file in files:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            temp_paths.append(tmp.name)

    try:
        processor = ImageProcessor()
        batch = processor(temp_paths)
        original_shapes = processor.original_shapes

        model = YOLO11(
            model_path="models/yolo11n.onnx",
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            filter_classes=filter_list,
        )
        detections_list = model(batch, original_shapes)

        results = []
        visualizer = BoundingBoxVisualization(class_names=model.class_names)

        for file, detections, img_path in zip(files, detections_list, temp_paths):
            original_image = cv2.imread(img_path)
            annotated = visualizer.run(original_image, detections.boxes)
            _, jpeg = cv2.imencode(".jpg", annotated)
            image_base64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
            boxes_data = [box.tolist() for box in detections.boxes]
            results.append(
                {"filename": file.filename, "image": image_base64, "detections": boxes_data}
            )

        return JSONResponse(content={"results": results})

    finally:
        for path in temp_paths:
            os.unlink(path)
