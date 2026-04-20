# CV Detection Platform

FastAPI + ONNX Runtime + MySQL based object detection API.

## Prerequisites:
# - MySQL (installed and running)
# - Python 3.10+

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/aliaksandrmarchuk369/Computer-Vision-Platform.git
cd cv-detection-platform
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
```

#### On Windows:
```bash
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -e .
pip install -e ".[dev]"  # optional, for testing/linting
```

### 4. Export YOLO model to ONNX

```bash
python scripts/export.py
```

### 5. Run the FastAPI server
```bash
uvicorn main:app --reload
```

## Demo
![Detection results](screenshots/demo.png)