import torch
import torchvision

class Extractor(torch.nn.Module):
    def __init__(self, model_path, embed_size=512):
        super().__init__()
        self.embed_size = embed_size
        self.model = torchvision.models.resnet101()
        self.extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        checkpoint = torch.load(model_path, weights_only=True)
        self.extractor.load_state_dict(checkpoint)
        self.fc = torch.nn.Linear(2048, self.embed_size)
        self.batch_norm = torch.nn.BatchNorm1d(self.embed_size)

    def forward(self, img):
        img = self.extractor(img)
        img = img.view(img.size(0), -1)
        img = self.fc(img)
        img = self.batch_norm(img)
        return torch.nn.functional.normalize(img, p=2, dim=1)