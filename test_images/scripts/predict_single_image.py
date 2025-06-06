from ultralytics import YOLO
import cv2
import os

# Load model
model_path = 'runs/classify/train2/weights/best.pt'
model = YOLO(model_path)

# Load ảnh cần dự đoán
image_path = 'test_images/lanhot1.jpg'  # KHÔNG cần ../ nếu chạy đúng từ tree_classification_project/

img = cv2.imread(image_path)

if img is None:
    print(f"❌ Không đọc được ảnh {image_path}")
    exit()

# Predict
results = model.predict(img)
pred_label = results[0].probs.top1
label_name = model.names[pred_label]
confidence = results[0].probs.top1conf

# Vẽ khung viền
h, w, _ = img.shape
color = (0, 255, 0)
thickness = 4
cv2.rectangle(img, (10, 10), (w-10, h-10), color, thickness)

text = f"{label_name} ({confidence*100:.2f}%)"
cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

# Hiển thị
cv2.imshow("Ket qua du doan", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lưu kết quả
save_path = os.path.join('results', os.path.basename(image_path))
cv2.imwrite(save_path, img)
print(f"✅ Đã lưu kết quả tại {save_path}")
