from ultralytics import YOLO
import cv2
import os
import numpy as np
import math

# Load YOLOv8 detection model
model_path = 'runs/detect/leaf_yolov87/weights/best.pt'
model = YOLO(model_path)

# Load test images
test_folder = 'test_images'
images = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.png'))]

# Dự đoán
results = []
for file in images:
    path = os.path.join(test_folder, file)
    img = cv2.imread(path)
    if img is None:
        continue

    preds = model.predict(img)[0]

    if len(preds.boxes) > 0:
        box = preds.boxes[0]
        label_id = int(box.cls[0].item())
        label_name = model.names[label_id]
        conf = float(box.conf[0])
    else:
        label_name = "Khong nhan dang"
        conf = 0.0

    results.append((file, img, label_name, conf))

# Cấu hình hiển thị
images_per_page = 10
pages = math.ceil(len(results) / images_per_page)
current_page = 0
font = cv2.FONT_HERSHEY_SIMPLEX

def draw_page(page_idx):
    start = page_idx * images_per_page
    end = start + images_per_page
    items = results[start:end]

    canvas = np.ones((600, 1300, 3), dtype=np.uint8) * 255
    for i, (name, img, label, conf) in enumerate(items):
        img_resized = cv2.resize(img, (224, 224))
        cv2.rectangle(img_resized, (2, 2), (222, 222), (0, 255, 255), 3)

        text = f"{label} ({conf*100:.1f}%)" if conf > 0 else f"{label}"
        cv2.putText(img_resized, text, (5, 30), font, 0.7, (0, 0, 255), 2)

        row = i // 5
        col = i % 5
        x_offset = 20 + col * 255
        y_offset = 20 + row * 255
        canvas[y_offset:y_offset+224, x_offset:x_offset+224] = img_resized

    win = f"Trang {page_idx+1}/{pages}"
    cv2.imshow(win, canvas)
    return win

# Hiển thị trang đầu tiên
win = draw_page(current_page)

while True:
    key = cv2.waitKey(0)
    if key == ord('1') and current_page < pages - 1:
        cv2.destroyWindow(win)
        current_page += 1
        win = draw_page(current_page)
    elif key == ord('2') and current_page > 0:
        cv2.destroyWindow(win)
        current_page -= 1
        win = draw_page(current_page)
    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()
