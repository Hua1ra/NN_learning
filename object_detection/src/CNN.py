import torch

# CNN Class (based on the Tencent paper)
class CNN(torch.nn.Module):
    def __init__(self, output_channels=512):
        super().__init__()
        # Hyperparameters
        self.input_channel = 3
        self.output_channels = output_channels
        # Layers
        self.layer1 = self.create_layer(self.input_channel, 32)
        self.layer2 = self.create_layer(32, 64)
        self.layer3 = self.create_layer(64, 128)
        self.layer4 = self.create_layer(128, 256)
        self.layer5 = self.create_layer(256, self.output_channels)
    @staticmethod
    def create_layer(in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, # Convolutional filter
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=out_channels), # Batch normalization
            torch.nn.ReLU(), # Actiation function
            torch.nn.MaxPool2d(kernel_size=2, # Max pooling
                               stride=2),
            torch.nn.Dropout2d(p=0.25) # Dropout
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x