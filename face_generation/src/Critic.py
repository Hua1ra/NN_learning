import torch

class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = self.create_layer(in_channels=4,
                                        out_channels=32,
                                        is_input=True)
        self.layer2 = self.create_layer(in_channels=32,
                                        out_channels=64)
        self.layer3 = self.create_layer(in_channels=64,
                                        out_channels=128)
        self.layer4 = self.create_layer(in_channels=128,
                                        out_channels=256)
        self.layer5 = self.create_layer(in_channels=256,
                                        out_channels=512)
        self.layer6 = self.create_layer(in_channels=512,
                                        out_channels=1,
                                        is_output=True)
        self.flags_projection = torch.nn.Linear(7, 128 * 128)

    @staticmethod
    def create_layer(in_channels, out_channels, is_output=False, is_input=False):
        # No InstanceNorm and no ReLU on the last layer.
        # No InstanceNorm on the first layer
        if not is_output and not is_input:
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1),
                torch.nn.InstanceNorm2d(out_channels, affine=True),
                torch.nn.LeakyReLU(0.2)
            )
        if not is_output:
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1),
                torch.nn.LeakyReLU(0.2)
            )
        return torch.nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=4,
                                         stride=1,
                                         padding=0)

    def forward(self, img, flags):
        batch_size = img.size(0)
        # Project flags into a 4th channel
        flags = self.flags_projection(flags).view(batch_size, 1, 128, 128)
        img = torch.cat([img, flags], dim=1)
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        img = self.layer5(img)
        output = self.layer6(img)
        return output