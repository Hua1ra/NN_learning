import torch

# Positional Eencoding Class
class Embedding(torch.nn.Module):
    def __init__(self, num_tokens=20*20, embedding_size=256):
        super().__init__()
        # For each picture we have shape=(20, 20, 256) tensor of features. We also need to add dim for batch size
        self.positional_embedding = torch.nn.Parameter(torch.randn(1, num_tokens, embedding_size))
    def forward(self, x):
        return x + self.positional_embedding