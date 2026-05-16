import torch

# Multihed Attention Class (Based on ViT)
# We do not have the mask here 'cause our predictions do not depend on previous predictoins
class MultiheadAttention(torch.nn.Module):
    def __init__(self, num_heads=8, embed_dim=256):
        super().__init__()
        self.num_heads = num_heads # Number of attention heads
        self.head_dim = embed_dim // self.num_heads # Dimensionality of each head
        assert embed_dim % self.num_heads == 0 # Check if possible
        self.q = torch.nn.Linear(embed_dim, embed_dim) # Matrix to get Query for all heads
        self.k = torch.nn.Linear(embed_dim, embed_dim) # Matrix to get Keys for all heads
        self.v = torch.nn.Linear(embed_dim, embed_dim) # Matrix to get Values for all heads
        self.out = torch.nn.Linear(embed_dim, embed_dim) # Final output projection layer
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size() # x: shape=(batch_size, seq_len=400, embed_dim=256)
        # Get Queries, Keys, Values for all heads
        q = self.q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # q, k, v: shape=(batch_size, num_heads=8, seq_len=400, head_dim=32)
        # Get weights for context vector
        attention_logits = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_logits, dim=-1)
        # Get context vector
        attention = torch.matmul(attention_weights, v).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, embed_dim)
        # Final projection
        attention = self.out(attention)
        return attention