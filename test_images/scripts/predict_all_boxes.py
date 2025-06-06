from ultralytics import YOLO
import cv2
import os

# Load mô hình YOLOv8 đã huấn luyện
model = YOLO("runs/detect/leaf_finetuned/weights/best.pt")


# Thư mục test
test_folder = 'test_images'
save_folder = 'results_all_boxes'
os.makedirs(save_folder, exist_ok=True)

# Duyệt qua từng ảnh
for file in os.listdir(test_folder):
    if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    path = os.path.join(test_folder, file)
    img = cv2.imread(path)
    if img is None:
        continue

    # Dự đoán với ngưỡng confidence thấp để không bỏ sót
    results = model.predict(source=img, conf=0.1)[0]

    boxes = results.boxes
    if boxes is not None and len(boxes) > 0:
        for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} ({conf*100:.1f}%)"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(img, "Khong nhan dang", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Lưu ảnh kết quả
    save_path = os.path.join(save_folder, file)
    cv2.imwrite(save_path, img)
    print(f"✅ {file} -> {save_path}")

print("🎯 Đã xử lý xong tất cả ảnh test!")
