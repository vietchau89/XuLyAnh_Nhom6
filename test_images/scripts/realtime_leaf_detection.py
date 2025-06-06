import cv2
from ultralytics import YOLO
import numpy as np

# Load mô hình đã fine-tune
model = YOLO("runs/detect/leaf_finetuned/weights/best.pt")

# Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không thể mở webcam.")
    exit()

# Thiết lập tên cửa sổ
cv2.namedWindow("Leaf Detection - Realtime", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc frame từ webcam.")
        break

    # Làm mượt ảnh để giảm nhiễu
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Dự đoán với confidence thấp để không bỏ sót
    results = model.predict(source=blurred, conf=0.15, verbose=False)[0]
    boxes = results.boxes
    names = model.names

    if boxes is not None and len(boxes) > 0:
        for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(cls)]
            confidence = float(conf)

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Khong nhan dang duoc", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Hiển thị kết quả
    cv2.imshow("Leaf Detection - Realtime", frame)

    # Nhấn ESC để thoát
    key = cv2.waitKey(1)
    if key == 27:
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
