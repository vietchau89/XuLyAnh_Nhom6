from ultralytics import YOLO

model = YOLO("runs/detect/leaf_yolov87/weights/best.pt")
model.train(
    data="leaf_data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="leaf_yolov88"
)
