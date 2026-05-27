import math
import torch

class ArcFace(torch.nn.Module):
    def __init__(self, in_features, num_classes, s=64.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.vector = torch.nn.Parameter(torch.FloatTensor(num_classes, in_features))
        torch.nn.init.xavier_uniform_(self.vector)
        self.cosm = math.cos(m)
        self.sinm = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_vector, label):
        cos = torch.nn.functional.linear(torch.nn.functional.normalize(input_vector), torch.nn.functional.normalize(self.vector))
        sin = torch.sqrt((1.0 - torch.pow(cos, 2))).clamp(0, 1)
        angle = cos * self.cosm - sin * self.sinm
        angle = torch.where(cos > self.th, angle, cos - self.mm)
        label_vector = torch.zeros_like(cos)
        label_vector.scatter_(1, label.view(-1, 1).long(), 1)
        output_vector = (label_vector * angle) + ((1.0 - label_vector) * cos)
        output_vector *= self.s
        return torch.nn.functional.cross_entropy(output_vector, label)