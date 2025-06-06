import os
import random
import shutil

def split_dataset(source_dir, dest_dir, train_ratio=0.8):
    # Tạo folder train và val
    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Duyệt từng lớp
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Tạo folder cho lớp trong train và val
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        # Copy ảnh vào train
        for img in train_imgs:
            if img.endswith(('.jpg', '.jpeg', '.png')):  # Chỉ copy file ảnh
                shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

        # Copy ảnh vào val
        for img in val_imgs:
            if img.endswith(('.jpg', '.jpeg', '.png')):  # Chỉ copy file ảnh
                shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

   

    print("✅ Dataset đã được chia thành train/val thành công!")

if __name__ == "__main__":
    source_folder = "Tree"        # Folder ảnh gốc của bạn
    destination_folder = "dataset"  # Folder mới sẽ lưu train/val
    split_dataset(source_folder, destination_folder, train_ratio=0.8)
