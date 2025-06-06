from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='leaf_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='runs/detect',
    name='leaf_yolov8',
    exist_ok=True
)
