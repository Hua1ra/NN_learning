import torch

# Feed Forward Class
class FeedForward(torch.nn.Module):
    def __init__(self, embed_dim=256, expansion=4):
        super().__init__()
        self.fc1 = torch.nn.Linear(embed_dim, expansion * embed_dim) # Expand output (256 -> 1024)
        self.gelu = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(expansion * embed_dim, embed_dim) # Back to embed_dim (1024 -> 256)
        self.dropout = torch.nn.Dropout(p=0.1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x