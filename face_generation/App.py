import os
import torch
import torch_directml
import tkinter as tk
from PIL import Image, ImageTk
from src.Generator import Generator
from tkinter import ttk, filedialog

class GANGeneratorTkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WGAN-GP Face Generator")
        self.root.geometry("600x450")
        self.root.resizable(False, False)
        self.model = None
        self.device = torch_directml.device(0)
        self.models_dir = "./models/"
        self.z = torch.randn(1, 100).to(self.device)
        self.attr_names = [
            'Male',
            'Young',
            'Pale Skin',
            'Bald',
            'Mustache',
            'Eyeglasses',
            'Smiling'
        ]
        self.model_selector = None
        self.right_panel = None
        self.left_panel = None
        self.image_label = None
        self.btn_random = None
        self.btn_save = None
        self.tk_image = None
        self.pil_image = None
        self.checkbox_vars = []
        self.init_ui()
        self.load_selected_model()

    def init_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)
        # Left panel
        self.left_panel = ttk.Frame(main_frame, width=300)
        self.left_panel.pack(side='left', fill='y', padx=(0, 10))
        # Model selection
        model_group = ttk.LabelFrame(self.left_panel, text=" Choose the model ", padding="10")
        model_group.pack(fill='x', pady=(0, 10))
        self.model_selector = ttk.Combobox(model_group, state="readonly")
        self.refresh_model_list()
        self.model_selector.pack(fill='x')
        self.model_selector.bind("<<ComboboxSelected>>", lambda e: self.load_selected_model())
        # Attributes
        attr_group = ttk.LabelFrame(self.left_panel, text=" Attributes ", padding="10")
        attr_group.pack(fill='x', pady=(0, 10))
        for name in self.attr_names:
            var = tk.DoubleVar()
            cb = ttk.Checkbutton(attr_group, text=name, variable=var, command=self.generate_face)
            cb.pack(anchor='w', pady=2)
            self.checkbox_vars.append(var)
        # Regenerate noize tensor
        self.btn_random = ttk.Button(self.left_panel, text="Generate image",
                                     command=self.regenerate_latent_vector)
        self.btn_random.pack(fill='x', pady=10)
        # Save button
        self.btn_save = ttk.Button(self.left_panel, text="Save Image",
                                   command=self.save_face)
        self.btn_save.pack(fill='x', pady=5)
        # Right panel
        self.right_panel = ttk.Frame(main_frame)
        self.right_panel.pack(side='right', fill='both', expand=True)
        self.image_label = ttk.Label(self.right_panel, text="Photo", anchor='center')
        self.image_label.pack(fill='both', expand=True)

    def refresh_model_list(self):
        models = [f for f in os.listdir(self.models_dir) if f.endswith('.pth')]
        models.sort()
        self.model_selector['values'] = models
        if models:
            self.model_selector.current(0)
        if not self.model_selector['values']:
            self.model_selector['values'] = ["No checkpoints found!"]
            self.model_selector.current(0)

    def load_selected_model(self):
        model_name = self.model_selector.get()
        if model_name == "No checkpoints found!":
            return
        model_path = self.models_dir + model_name
        try:
            self.model = Generator()
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['generator'])
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
           self.image_label.config(text=f"Loading Error:\n{str(e)}")

    def regenerate_latent_vector(self):
        self.z = torch.randn(1, 100).to(self.device)
        self.generate_face()

    def generate_face(self):
        if self.model is None:
            return
        # Get checkbox values
        flags_list = [var.get() for var in self.checkbox_vars]
        flags_tensor = torch.tensor([flags_list]).to(self.device)
        with torch.no_grad():
            fake_tensor = self.model(self.z, flags_tensor).cpu()
            fake_tensor = (fake_tensor + 1.0) / 2.0
            fake_tensor = fake_tensor.clamp(0.0, 1.0) * 255.0
            img_np = fake_tensor.squeeze(0).permute(1, 2, 0).byte().numpy()
        # PIL Image from the numpy array
        self.pil_image = Image.fromarray(img_np)
        self.pil_image = self.pil_image.resize((256, 256))
        # Convert to a proper format
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        self.image_label.config(image=self.tk_image, text="")

    def save_face(self):
        if self.pil_image is None:
            return
        # Get the path to save
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")],
            title="Save generated face as..."
        )
        # Try to save
        if file_path:
            try:
                self.pil_image.save(file_path)
            except Exception as e:
                self.image_label.config(text=f"Save Error:\n{str(e)}")



if __name__ == "__main__":
    root_tk = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    app = GANGeneratorTkApp(root_tk)
    root_tk.mainloop()