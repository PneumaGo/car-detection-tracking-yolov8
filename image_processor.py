import cv2
from .model_loader import load_model
from .config import TEMP_OUTPUT_IMAGE

model = load_model()

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return None, "Error: Could not load image."

    results = model.predict(frame, iou=0.5, conf=0.4, imgsz=480, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imwrite(TEMP_OUTPUT_IMAGE, annotated_frame)
    return TEMP_OUTPUT_IMAGE, "Counting is not applied to images."
