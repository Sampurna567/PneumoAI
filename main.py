main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from model import predict_pneumonia, generate_gradcam
import cv2

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#NEW

from fastapi.staticfiles import StaticFiles

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label, confidence = predict_pneumonia(file_path)

    
    heatmap_path = generate_gradcam(file_path)

    return {
        "prediction": "Pneumonia" if label == "PNEUMONIA" else "Normal",
        "confidence": round(confidence * 100, 2),
        "image_path": f"http://127.0.0.1:8000/uploads/{file.filename}",
        "heatmap": f"http://127.0.0.1:8000/uploads/{os.path.basename(heatmap_path)}"
    }
