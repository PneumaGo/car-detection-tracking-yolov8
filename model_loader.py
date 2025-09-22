from ultralytics import YOLO
import torch
from .config import MODEL_PATH

def load_model():
    try:
        model = YOLO(MODEL_PATH)
        model.fuse()
        if torch.cuda.is_available():
            model.to(torch.device('cuda'))
            model.half()
            print("Model loaded on GPU with FP16.")
        else:
            print("GPU not found, model will run on CPU.")
    except Exception as e:
        print(f"Error loading custom YOLOv8 model: {e}")
        model = YOLO('yolov8n.pt')
        model.fuse()
        print("Loaded standard YOLOv8n model as fallback.")
    return model
