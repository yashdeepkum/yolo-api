from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi.responses import JSONResponse
import io

app = FastAPI()
model = YOLO("yolov8naircraft.pt")  # Your YOLOv8n model

@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    # Run detection
    results = model(image)[0]

    # Extract detections
    detections = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "class": model.names[class_id],
            "confidence": round(confidence, 3),
            "box": [x1, y1, x2, y2]
        })

    return JSONResponse(content={"detections": detections})
