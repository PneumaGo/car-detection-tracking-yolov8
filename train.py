from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='my_dataset_yolo/data.yaml', epochs=100, imgsz=640, model='yolov8m.pt')