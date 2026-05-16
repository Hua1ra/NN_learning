import torch
from .FeedForward import FeedForward
from .MultiHeadAttention import MultiheadAttention

# Transformer Block Class
class TransformerBlock(torch.nn.Module):
    def __init__(self, num_heads=8, embed_dim=256, expansion=4, normalized_shape=256):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(normalized_shape=normalized_shape)
        self.multihead_attention = MultiheadAttention(num_heads=num_heads, embed_dim=embed_dim)
        self.ln2 = torch.nn.LayerNorm(normalized_shape=normalized_shape)
        self.feed_forward = FeedForward(embed_dim=embed_dim, expansion=expansion)
    def forward(self, x):
        # Transformer architecture with MHA, FF, LN and ResNet parts
        residual = x
        x = self.ln1(x)
        x = self.multihead_attention(x)
        x = x + residual
        residual = x
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = x + residual
        return x