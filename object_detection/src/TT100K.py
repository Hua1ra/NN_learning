import json
import torch
from PIL import Image
from torchvision import transforms

# Now we can create a Dataset class
class TT100KDataset(torch.utils.data.Dataset):
    def __init__(self, base_path='./resources/tt100k_2021/train',
                 labels_path='./resources/tt100k_2021/labels_all.json',
                 transformer=None):
        super().__init__()
        self.base_path = base_path # Base path to all the pictures
        if transformer is None: # Redeclare image transformer if it is None
            self.transformer = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transformer = transformer
        with open(labels_path, 'r') as j:
            self.labels_dict = json.load(j)
    def __len__(self): # Dataset length
        return len(self.labels_dict)
    def __getitem__(self, idx): # Get image and labels 'on the fly'
        img = Image.open(f'{self.base_path}/{idx}.jpg')
        labels = torch.tensor(self.labels_dict[f'{idx}'], dtype=torch.float32)
        img = self.transformer(img)
        return img, labels