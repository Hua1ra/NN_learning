import torch
from .CNN import CNN
from .Embedding import Embedding
from .TransformerBlock import TransformerBlock

# Actual Sigh Detection Model
class SignDetectionModel(torch.nn.Module):
    def __init__(self, output_channels=512, num_heads=8, embed_dim=256, expansion=4, normalized_shape=256, num_transformer_layers=8):
        super().__init__()
        self.tiles = 20
        self.items_per_tile = 5
        self.output_channels = output_channels
        self.CNN = CNN(output_channels=output_channels)
        self.transformer = torch.nn.Sequential(
            *[TransformerBlock(num_heads=num_heads,
                               embed_dim=embed_dim,
                               expansion=expansion,
                               normalized_shape=normalized_shape) for _ in range(num_transformer_layers)]
        )
        self.CNN_projection = torch.nn.Linear(output_channels, embed_dim)
        self.embedding = Embedding(num_tokens=self.tiles * self.tiles,
                                   embedding_size=embed_dim)
        self.head = torch.nn.Linear(in_features=embed_dim,
                                    out_features=self.items_per_tile)
    def forward(self, x):
        # x: shape=(batch_size, output_channels=512, tiles=20, tiles=20)
        x = self.CNN(x)
        # transform x to shape=(batch_size, tiles=20, tiles=20, output_channels=512)
        x = x.permute(0, 2, 3, 1).view(-1, self.tiles * self.tiles, self.output_channels)
        # x: shape=(batch_size, tiles^2=400, output_channel=512)
        x = self.CNN_projection(x)
        # x: shape=(batch_size, tiles^2=400, embed_dim=256)
        # Add positional embedding
        x = self.embedding(x)
        # Transformer
        x = self.transformer(x)
        # Regression head
        x = self.head(x)
        return x