import io
import pickle
import torchvision
from PIL import Image

class ValDataset:
    def __init__(self, root, mode='transformer', transformer=None):
        self.root = root
        self.mode = mode
        self.transformer = transformer
        if self.transformer is None:
            self.transformer = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])
        with open(self.root, 'rb') as f:
            self.bins, self.labels = pickle.load(f, encoding='bytes')


    def __getitem__(self, item):
        label = self.labels[item]
        img1 = self.bins[2 * item]
        img2 = self.bins[2 * item + 1]
        img1 = Image.open(io.BytesIO(img1)).convert('RGB')
        img2 = Image.open(io.BytesIO(img2)).convert('RGB')
        if self.mode == 'transformer':
            img1 = self.transformer(img1)
            img2 = self.transformer(img2)
        return img1, img2, label


    def __len__(self):
        return len(self.labels)

    def set_mode(self, mode):
        self.mode = mode

    def get_mode(self):
        return self.mode