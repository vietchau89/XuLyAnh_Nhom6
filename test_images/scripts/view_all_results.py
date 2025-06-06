import os
import cv2
import numpy as np
import math

# Thư mục chứa ảnh kết quả đã vẽ box
folder = 'results_all_boxes'
images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
images.sort()

# Cấu hình hiển thị
images_per_page = 10
cols = 5
rows = 2
img_size = 224
padding = 20
font = cv2.FONT_HERSHEY_SIMPLEX

# Tính tổng số trang
total_images = len(images)
total_pages = math.ceil(total_images / images_per_page)
current_page = 0

def draw_page(page_idx):
    start = page_idx * images_per_page
    end = min(start + images_per_page, total_images)
    items = images[start:end]

    canvas_width = cols * (img_size + padding) + padding
    canvas_height = rows * (img_size + padding) + padding
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    for i, img_name in enumerate(items):
        path = os.path.join(folder, img_name)
        img = cv2.imread(path)
        if img is None:
            continue
        img_resized = cv2.resize(img, (img_size, img_size))

        row = i // cols
        col = i % cols
        x_offset = padding + col * (img_size + padding)
        y_offset = padding + row * (img_size + padding)

        canvas[y_offset:y_offset+img_size, x_offset:x_offset+img_size] = img_resized
        cv2.putText(canvas, img_name, (x_offset, y_offset - 5), font, 0.5, (0, 0, 0), 1)

    window_title = f"Trang {page_idx+1}/{total_pages} - {folder}"
    cv2.imshow(window_title, canvas)
    return window_title

# Hiển thị trang đầu tiên
window = draw_page(current_page)

# Duyệt trang bằng phím 1 và 2
while True:
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break
    elif key == ord('1') and current_page < total_pages - 1:
        cv2.destroyWindow(window)
        current_page += 1
        window = draw_page(current_page)
    elif key == ord('2') and current_page > 0:
        cv2.destroyWindow(window)
        current_page -= 1
        window = draw_page(current_page)

cv2.destroyAllWindows()
