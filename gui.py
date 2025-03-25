import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import os

class ZoomableAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Zoomable Multi-Class Annotator")

        self.canvas = tk.Canvas(root, cursor="cross", bg="gray")
        self.canvas.pack(fill="both", expand=True)

        self.menu = tk.Menu(root)
        self.root.config(menu=self.menu)

        file_menu = tk.Menu(self.menu, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Save Mask", command=self.save_mask)
        self.menu.add_cascade(label="File", menu=file_menu)

        # Data
        self.image = None
        self.original_image = None
        self.tk_img = None
        self.mask = None
        self.original_mask = None
        self.radius = 6
        self.current_class = 1
        self.zoom = 1.0
        self.zoom_step = 0.1

        self.colors = {
            1: "red",
            2: "green",
            3: "blue",
            4: "yellow",
            5: "magenta"
        }

        # Events
        self.canvas.bind("<B1-Motion>", self.draw_on_mask)
        self.canvas.bind("<Button-4>", self.zoom_in)   # Linux scroll up
        self.canvas.bind("<Button-5>", self.zoom_out)  # Linux scroll down
        self.canvas.bind("<MouseWheel>", self.mouse_wheel_zoom)  # Windows/Mac
        self.root.bind("<Key>", self.change_class)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png")])
        if not path:
            return

        self.image_path = path
        self.original_image = Image.open(path).convert("RGB")
        self.original_mask = np.zeros(self.original_image.size[::-1], dtype=np.uint8)
        self.zoom = 1.0
        self.update_view()

    def update_view(self):
        w, h = self.original_image.size
        new_size = (int(w * self.zoom), int(h * self.zoom))
        resized = self.original_image.resize(new_size, Image.NEAREST)
        self.tk_img = ImageTk.PhotoImage(resized)

        self.mask = cv2.resize(
            self.original_mask,
            dsize=new_size[::-1],
            interpolation=cv2.INTER_NEAREST
        )

        self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def draw_on_mask(self, event):
        if self.tk_img is None:
            return

        x, y = int(event.x), int(event.y)
        color = self.colors.get(self.current_class, "red")
        self.canvas.create_oval(
            x - self.radius, y - self.radius,
            x + self.radius, y + self.radius,
            fill=color, outline=color, width=0
        )

        # Draw on resized mask
        cv2.circle(self.mask, (x, y), self.radius, color=self.current_class, thickness=-1)

        # Update the full-resolution mask too
        scale = 1 / self.zoom
        orig_x, orig_y = int(x * scale), int(y * scale)
        orig_radius = max(1, int(self.radius * scale))
        cv2.circle(self.original_mask, (orig_x, orig_y), orig_radius, color=self.current_class, thickness=-1)

    def mouse_wheel_zoom(self, event):
        if event.delta > 0:
            self.zoom_in(event)
        else:
            self.zoom_out(event)

    def zoom_in(self, event=None):
        self.zoom *= (1 + self.zoom_step)
        self.update_view()

    def zoom_out(self, event=None):
        self.zoom /= (1 + self.zoom_step)
        self.update_view()

    def change_class(self, event):
        if event.char.isdigit():
            num = int(event.char)
            if num >= 0:
                self.current_class = num
                print(f"Switched to class {num}")

    def save_mask(self):
        if self.original_mask is None:
            return
        save_dir = filedialog.askdirectory(title="Select Mask Save Directory")
        if not save_dir:
            return
        filename = os.path.basename(self.image_path)
        mask_path = os.path.join(save_dir, f"mask_{filename}")
        cv2.imwrite(mask_path, self.original_mask)
        print(f"Saved mask to {mask_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ZoomableAnnotator(root)
    root.mainloop()
