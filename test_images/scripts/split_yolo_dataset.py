import os
import shutil
from sklearn.model_selection import train_test_split

# Đường dẫn thư mục nguồn
raw_folder = "raw_images"
image_files = [f for f in os.listdir(raw_folder) if f.endswith(".jpg")]

# Chia train/val (80/20)
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Thư mục đích
def copy_files(split, files):
    for img_name in files:
        name, _ = os.path.splitext(img_name)
        label_name = name + ".txt"

        # Copy ảnh
        shutil.copy(os.path.join(raw_folder, img_name), os.path.join(f"dataset/images/{split}", img_name))

        # Copy nhãn
        label_src = os.path.join(raw_folder, label_name)
        label_dst = os.path.join(f"dataset/labels/{split}", label_name)

        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            print(f"⚠️ Không tìm thấy nhãn: {label_name}")

# Copy dữ liệu
copy_files("train", train_files)
copy_files("val", val_files)

print("✅ Đã chia xong dữ liệu và copy vào dataset/")
