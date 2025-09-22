# car-detection-tracking-yolov8
# Vehicle detection, tracking, and counting using YOLOv8, BoT-SORT, and Gradio.

This project performs vehicle detection, tracking, and counting in videos using [YOLOv8](https://github.com/ultralytics/ultralytics), [BoT-SORT](https://arxiv.org/abs/2211.11164) tracker, and [Gradio](https://www.gradio.app/) for the user interface.

---

## Features
- Object detection on images and videos
- Object tracking with unique IDs
- Vehicle counting when crossing a predefined line
- Annotated video outputs with bounding boxes and counters
- Web interface powered by Gradio

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/car-detection-tracking-yolov8.git
cd car-detection-tracking-yolov8
pip install -r requirements.txt
````

---

## Project Structure

```
car-detection-tracking-yolov8/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── config.py              # Global configuration
│   ├── model_loader.py        # Model loading
│   ├── utils.py               # Utility functions
│   ├── image_processor.py     # Image processing
│   ├── video_processor.py     # Video processing and counting
│   ├── gradio_app.py          # Gradio app logic
│   ├── main.py                # Entry point
│   └── train.py               # Training script for YOLOv8
```

### `train.py` Example

```python
from ultralytics import YOLO

# Load a pretrained model (recommended for training)
model = YOLO('yolov8m.pt')

# Train the model
results = model.train(
    data='my_dataset_yolo/data.yaml',
    epochs=100,
    imgsz=640,
    model='yolov8m.pt'
)
```

---

## Dataset Preparation

The custom dataset was prepared as follows:

1. **Extract frames from video**
   Frames were extracted using `ffmpeg`:

   ```bash
   ffmpeg -i train.mp4 -vf fps=1.5 images_train/1_img_%03d.jpg
   ffmpeg -i val.mp4 -vf fps=1.5 images_val/1_img_%03d.jpg
   ```

   `fps=5` means 1.5 frames per second were saved.

    https://drive.google.com/drive/folders/1_b0UCabCyHdtc4Kg5O3WxbrnEF4_KL8-?usp=sharing
   
    
3. **Image annotation**

   * CVAT was used for annotation.

   * CVAT ran locally via Docker:

     ```bash
     docker-compose up -d
     ```

   * Annotation was done in the CVAT web interface and exported in YOLO format.
  
     After exporting from CVAT, need to reorganize the dataset and create a proper data.yaml before training in YOLOv8.
     
     CVAT YOLO 1.1 may include extra fields (class_id, frame_id) that YOLOv8 does not support.
     https://drive.google.com/drive/folders/1oDnWYz9InvcJQhJm4hgchVgCQ9GowvvL?usp=sharing
     
     YOLOv8 requires a different dataset structure (images/train, labels/train, etc.) and a custom data.yaml
     https://drive.google.com/drive/folders/1sPUDIgf3a8Z67YhpytAzotv_K-OihMKr?usp=sharing
     

4. **Training YOLOv8**
   The annotated dataset was used to train a YOLOv8 model as shown in `train.py`.

---

## Dataset and Model Limitations

* Dataset is relatively small train(\~90 images) val(\~30)
* Two classes: `car` and `van`
* `Van` class has fewer samples → weaker generalization for this class
* Model suitable for demonstration; larger datasets needed for production-ready performance

---

## Training Notes

* **Backbone:** YOLOv8 (pretrained, fine-tuned on custom dataset)
* **Epochs:** 100
* **Image size:** 640
* **Goal:** Demonstrate training pipeline and inference
* **Limitation:** Small dataset, results not suitable for production-level accuracy

---

## Requirements

* Python 3.8+
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* Gradio
* NumPy

All dependencies are listed in **`requirements.txt`**.

---

## Usage

Run the main entry point:

```bash
python src/main.py
```

* Upload an **image** → detection only
* Upload a **video** → detection, tracking, and vehicle counting

The app displays:

* Annotated outputs (bounding boxes, IDs, counting line)
* Class-specific count summary

---

## Example and Demonstration on Hugging Face

**Input video:**

A live demo of the model is available on [Hugging Face Spaces](https://huggingface.co/spaces/PneumaGo/car-counter-demo).  

Please note:
- When running for the first time, the model weights will be downloaded automatically.  
- This may take some time depending on your internet connection.  
- After loading, you can test the model by uploading an image (for detection only) or a video (for detection, tracking, and counting).  

**Output:**

* Annotated video with bounding boxes, IDs, and counting line
* Text summary with counts per class

---

## Future Improvements

* Add bidirectional counting (entry/exit)
* Integrate with a database for saving statistics
* Deploy as a real-time web service (FastAPI / Flask)

---

## License

MIT License

```
