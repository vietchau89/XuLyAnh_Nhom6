import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import os

# Constants
IMAGES_PER_PAGE = 6
IMAGES_PER_ROW = 3
IMAGE_SIZE = (320, 240)
image_refs = []  # prevent garbage collection

class LeafViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 Trình xem kết quả dự đoán lá cây")
        self.folder = None
        self.current_page = 0
        self.images = []

        self.setup_ui()

    def setup_ui(self):
        self.root.geometry("1200x700")
        self.root.configure(bg="white")

        self.main_frame = tk.Frame(self.root, bg="white")
        self.main_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.main_frame, bg="white")
        self.scrollbar = tk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = tk.Frame(self.canvas, bg="white")
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Right Panel
        self.side_panel = tk.Frame(self.root, bg="#f0f0f0", width=150)
        self.side_panel.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")

        self.choose_btn = tk.Button(self.side_panel, text="📂\nChọn thư mục", font=("Segoe UI", 10, "bold"),
                                     bg="#2196F3", fg="white", height=3, command=self.choose_folder)
        self.choose_btn.pack(pady=(30, 20), fill="x", padx=10)

        self.prev_btn = tk.Button(self.side_panel, text="←\nTrước", font=("Segoe UI", 10, "bold"),
                                   bg="#9E9E9E", fg="white", height=3, command=self.prev_page)
        self.prev_btn.pack(pady=10, fill="x", padx=10)

        self.next_btn = tk.Button(self.side_panel, text="→\nTiếp", font=("Segoe UI", 10, "bold"),
                                   bg="#4CAF50", fg="white", height=3, command=self.next_page)
        self.next_btn.pack(pady=10, fill="x", padx=10)

    def choose_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder = folder
            self.images = self.load_images()
            self.current_page = 0
            self.display_page()

    def load_images(self):
        loaded = []
        image_refs.clear()
        for filename in sorted(os.listdir(self.folder)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(self.folder, filename)
                img = Image.open(path).convert("RGB")
                draw = ImageDraw.Draw(img)

                txt_path = os.path.splitext(path)[0] + ".txt"
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls, x, y, w, h = map(float, parts)
                                W, H = img.size
                                x1 = int((x - w/2) * W)
                                y1 = int((y - h/2) * H)
                                x2 = int((x + w/2) * W)
                                y2 = int((y + h/2) * H)
                                draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
                                draw.text((x1, y1 - 18), f"{int(cls)}", fill="red")

                img_resized = img.resize(IMAGE_SIZE)
                img_tk = ImageTk.PhotoImage(img_resized)
                image_refs.append(img_tk)
                loaded.append((filename, img_tk))
        return loaded

    def display_page(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        start = self.current_page * IMAGES_PER_PAGE
        end = start + IMAGES_PER_PAGE
        page_images = self.images[start:end]

        for i, (filename, img) in enumerate(page_images):
            frame = tk.Frame(self.scrollable_frame, bg="white", bd=2, relief="groove")
            label = tk.Label(frame, image=img, text=filename, compound="top", font=("Segoe UI", 9, "bold"))
            label.pack(padx=10, pady=10)
            row, col = divmod(i, IMAGES_PER_ROW)
            frame.grid(row=row, column=col, padx=20, pady=20)

    def next_page(self):
        if (self.current_page + 1) * IMAGES_PER_PAGE < len(self.images):
            self.current_page += 1
            self.display_page()

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.display_page()


if __name__ == "__main__":
    root = tk.Tk()
    app = LeafViewerApp(root)
    root.mainloop()
