import torch
import torchvision
from ultralytics import YOLO

class Detector(torch.nn.Module):
    def __init__(self, model_path, transformer=None):
        super().__init__()
        self.detector = YOLO(model_path)
        self.transformer = transformer
        self.mode = 'tensor'
        if self.transformer is None:
            self.transformer = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])

    def forward(self, img):
        results = self.detector(img, verbose=False)
        if not results:
            return None
        x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
        img = img.crop((x1, y1, x2, y2))
        if self.mode == 'tensor':
            img = self.transformer(img).unsqueeze(0)
        return img

    def detect(self):
        self.mode = 'detect'

    def tensor(self):
        self.mode = 'tensor'