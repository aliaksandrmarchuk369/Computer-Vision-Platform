import base64
import json
import os
import tempfile
import time
from contextlib import asynccontextmanager

import cv2
import mlflow
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from starlette.templating import _TemplateResponse

from src.bounding_box_visualization import BoundingBoxVisualization
from src.db import (
    clear_all_detections,
    create_table,
    ensure_database,
    get_gallery_detections,
    get_history_detections,
    insert_detection,
)
from src.image_processor import ImageProcessor
from src.yolo11 import YOLO11

# Set MLflow tracking URI (optional, defaults to ./mlruns)
mlflow.set_tracking_uri("file:./mlruns")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure database and table exist
    ensure_database()
    create_table()
    yield
    # Shutdown: nothing to clean up


app = FastAPI(lifespan=lifespan)

# Jinja2 setup
env = Environment(loader=FileSystemLoader("templates"), autoescape=True, cache_size=0)

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")


def template_response(name: str, context: dict):
    template = env.get_template(name)
    return _TemplateResponse(template, context)


@app.get("/")
async def home(request: Request):
    return template_response("index.html", {"request": request})


@app.get("/history")
async def history(request: Request):
    records = get_history_detections()
    return template_response("history.html", {"request": request, "records": records})


@app.delete("/history/clear")
async def clear_history():
    clear_all_detections()
    return JSONResponse(content={"message": "History cleared successfully"})


@app.get("/gallery")
async def gallery(request: Request):
    records = get_gallery_detections()
    return template_response("gallery.html", {"request": request, "records": records})


@app.post("/detect")
async def detect(
    files: list[UploadFile] = File(...),
    conf_threshold: float = Query(0.3, ge=0.0, le=1.0),
    nms_threshold: float = Query(0.45, ge=0.0, le=1.0),
    filter_classes: str | None = Query(None, description="Comma-separated class names or IDs"),
):
    # Parse filter_classes (supports both names and IDs)
    filter_list = None
    if filter_classes:
        parts = [c.strip() for c in filter_classes.split(",")]
        if all(p.isdigit() for p in parts):
            filter_list = [int(p) for p in parts]
        else:
            filter_list = parts  # list of class names

    temp_paths = []
    for file in files:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            temp_paths.append(tmp.name)

    try:
        # Start MLflow run
        with mlflow.start_run(run_name="detection"):
            # Log parameters
            mlflow.log_params(
                {
                    "conf_threshold": conf_threshold,
                    "nms_threshold": nms_threshold,
                    "filter_classes": filter_classes or "all",
                }
            )

            # Measure inference time
            start_time = time.perf_counter()

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

            inference_time_ms = (time.perf_counter() - start_time) * 1000
            mlflow.log_metric("inference_time_ms", inference_time_ms)

            results = []
            visualizer = BoundingBoxVisualization(class_names=model.class_names)
            total_detections = 0

            for file, detections, img_path in zip(files, detections_list, temp_paths):
                original_image = cv2.imread(img_path)
                annotated = visualizer.run(original_image, detections.boxes)
                _, jpeg = cv2.imencode(".jpg", annotated)
                image_base64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
                boxes_data = [box.tolist() for box in detections.boxes]
                total_detections += len(boxes_data)
                results.append(
                    {"filename": file.filename, "image": image_base64, "detections": boxes_data}
                )

                # Insert into MySQL with annotated image
                insert_detection(
                    filename=file.filename,
                    num_detections=len(boxes_data),
                    detections_json=json.dumps(boxes_data),
                    conf_thr=conf_threshold,
                    nms_thr=nms_threshold,
                    filter_cls=filter_classes or "",
                    annotated_image=image_base64,
                )

            mlflow.log_metric("total_detections", total_detections)
            return JSONResponse(content={"results": results})

    finally:
        for path in temp_paths:
            os.unlink(path)
