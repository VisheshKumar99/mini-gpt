import torch

from src.attention import CausalSelfAttention


# --------------------------------
# Configuration
# --------------------------------

BATCH_SIZE = 2
CONTEXT_LENGTH = 8
EMBEDDING_DIM = 16 


# --------------------------------
# Fake embedding input
# --------------------------------

x = torch.randn(
    BATCH_SIZE,
    CONTEXT_LENGTH,
    EMBEDDING_DIM
)

print("Input:")
print(x.shape)


# --------------------------------
# Attention layer
# --------------------------------

attention = CausalSelfAttention(
    embedding_dim=EMBEDDING_DIM,
    context_length=CONTEXT_LENGTH
)


# --------------------------------
# Forward pass
# --------------------------------

output, attention_weights = attention(x)

print("Output:")
print(output.shape)
print("Attention weights:")
print(attention_weights.shape)

print(attention_weights[0])

print("\nRow sums:")
print(attention_weights[0].sum(dim=-1))