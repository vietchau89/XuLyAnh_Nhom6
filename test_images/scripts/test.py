import os

label_dir = 'dataset/labels/train'
empty_files = [f for f in os.listdir(label_dir) if os.path.getsize(os.path.join(label_dir, f)) == 0]

print(f"Có {len(empty_files)} file nhãn rỗng (không có bounding box):")
print(empty_files)
