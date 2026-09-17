import torch

from src.attention_multi import MultiHeadAttention


# --------------------------------
# Configuration
# --------------------------------

BATCH_SIZE = 2
CONTEXT_LENGTH = 8
EMBEDDING_DIM = 16
NUM_HEADS = 4


# --------------------------------
# Fake embedding input
# --------------------------------

x = torch.randn(
    BATCH_SIZE,
    CONTEXT_LENGTH,
    EMBEDDING_DIM
)


# --------------------------------
# Multi Head Attention
# --------------------------------

attention = MultiHeadAttention(
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    context_length=CONTEXT_LENGTH
)


# --------------------------------
# Forward pass
# --------------------------------

output, attention_weights = attention(x)


print("\n================================")
print("Final Results")
print("================================")

print("Output:")
print(output.shape)

print("\nAttention weights:")
print(attention_weights.shape)

print("\nFirst head attention matrix:")
print(attention_weights[0, 0])

print("\nRow sums:")
print(attention_weights[0, 0].sum(dim=-1))