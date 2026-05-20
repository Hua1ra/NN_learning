import torch

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = self.create_layer(in_channels=107,
                                        out_channels=512,
                                        is_input=True)
        self.layer2 = self.create_layer(in_channels=512,
                                        out_channels=256)
        self.layer3 = self.create_layer(in_channels=256,
                                        out_channels=128)
        self.layer4 = self.create_layer(in_channels=128,
                                        out_channels=64)
        self.layer5 = self.create_layer(in_channels=64,
                                        out_channels=32)
        self.layer6 = self.create_layer(in_channels=32,
                                        out_channels=3,
                                        is_output=True)
        self.tanh = torch.nn.Tanh()

    @staticmethod
    def create_layer(in_channels, out_channels, is_output=False, is_input=False):
        # No BatchNorm and no ReLU on the last layer
        if not is_output:
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels,
                                         out_channels,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1 if not is_input else 0), # size: 1 -> 4 on the first layer
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()
            )
        return torch.nn.ConvTranspose2d(in_channels,
                                         out_channels,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1)

    def forward(self, img, flags):
        # Concat noize tensor with flag tensor
        img = torch.cat([img, flags], dim=-1)
        batch_size = img.size(0)
        img = img.view(batch_size, -1, 1, 1)
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        img = self.layer5(img)
        img = self.layer6(img)
        output = self.tanh(img)
        return output