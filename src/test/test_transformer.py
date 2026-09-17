import torch

from src.transformer import TransformerBlock


BATCH_SIZE = 2
CONTEXT_LENGTH = 8
EMBEDDING_DIM = 16
NUM_HEADS = 4


# Fake token embeddings
x = torch.randn(
    BATCH_SIZE,
    CONTEXT_LENGTH,
    EMBEDDING_DIM
)


transformer = TransformerBlock(
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    context_length=CONTEXT_LENGTH
)


output, attention_weights = transformer(x)


print("\n================================")
print("Final Results")
print("================================")

print("\nInput:")
print(x.shape)

print("\nOutput:")
print(output.shape)

print("\nAttention weights:")
print(attention_weights.shape)

print("\nFirst head attention:")
print(attention_weights[0, 0])