import pandas as pd
import torch
import torchvision
from PIL import Image

class CelebA(torch.utils.data.Dataset):
    def __init__(self,
                 img_base_path='./resources/images_old/',
                 labels_base_path='./resources/list_attr_celeba.csv',
                 transformer=None):
        super().__init__()
        self.img_base_path = img_base_path
        self.labels_base_path = labels_base_path
        self.labels = pd.read_csv(labels_base_path, delimiter=',')
        self.labels.set_index('image_id', inplace=True)
        self.labels.replace(-1, 0, inplace=True)
        # Save only neccessary labels
        self.labels = self.labels[
            ['Male',
             'Young',
             'Pale_Skin',
             'Bald',
             'Mustache',
             'Eyeglasses',
             'Smiling']
        ]
        self.transformer = transformer
        if self.transformer is None:
            self.transformer = torchvision.transforms.Compose([
                torchvision.transforms.Resize((128, 128)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5]),
            ])
    def __getitem__(self, item):
        index = self.labels.index[item]
        img = self.transformer(Image.open(self.img_base_path + index).convert('RGB'))
        label = torch.Tensor(list(self.labels.iloc[item]))
        return img, label
    def __len__(self):
        return len(self.labels)