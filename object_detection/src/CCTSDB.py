import json
import torch
from PIL import Image
from torchvision import transforms

# Create validation dataset (the metrics can become worse due to format differences)
class CCTSDB(torch.utils.data.Dataset):
    def __init__(self,
                 base_path='./resources/CCTSDB/test/',
                 labels_path='./resources/CCTSDB/labels_all.json',
                 transformer=None):
        self.CNN_output = 20
        self.nums_per_pixel = 5
        self.base_path = base_path
        self.labels_path = labels_path
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
    def create_labels(self, image, objects):
        # Base tensor, which would be transformed into a label tensor
        labels = torch.zeros([self.CNN_output, self.CNN_output, self.nums_per_pixel], dtype=torch.float32)
        size = image.size
        for bbox in objects: # For each sign on the picture
            xmin = bbox['xmin']
            ymin = bbox['ymin']
            xmax = bbox['xmax']
            ymax = bbox['ymax']
            x_center = int(xmin + xmax) // 2 # Sign center coordinate x
            y_center = int(ymin + ymax) // 2 # Sign center coordinate x
            block_x = int(self.CNN_output / size[0] * x_center) # Block id (x)
            block_y = int(self.CNN_output / size[1] * y_center) # Block id (y)
            # Clamp values (0, CNN_output)
            block_x = max(0, block_x)
            block_y = max(0, block_y)
            block_x = min(self.CNN_output - 1, block_x)
            block_y = min(self.CNN_output - 1, block_y)
            # Relative coords
            x_center = (x_center % (size[0] / self.CNN_output)) / (size[0] / self.CNN_output)
            y_center = (y_center % (size[1] / self.CNN_output)) / (size[1] / self.CNN_output)
            width = (xmax - xmin) / size[0]
            height = (ymax - ymin) / size[1]
            labels[block_y, block_x] = torch.tensor([1, x_center, y_center, width, height])
        return labels.view(-1)
    def __len__(self):
        return len(self.labels_dict)
    def __getitem__(self, item):
        # Get transformed image with labels
        img_path = f'{str(item)}.png'
        img = Image.open(self.base_path + img_path).convert('RGB')
        labels = self.create_labels(img, self.labels_dict.get(str(item), []))
        img = self.transformer(img)
        return img, labels