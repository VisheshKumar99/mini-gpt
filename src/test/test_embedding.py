import torch

from src.embedding import GPTEmbedding


BATCH_SIZE = 2
CONTEXT_LENGTH = 8
VOCAB_SIZE = 100
EMBEDDING_DIM = 16


token_ids = torch.randint(
    0,
    VOCAB_SIZE,
    (BATCH_SIZE, CONTEXT_LENGTH)
)

print("Token IDs:")
print(token_ids)

embedding = GPTEmbedding(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    context_length=CONTEXT_LENGTH
)

output = embedding(token_ids)

print("\nToken IDs shape:")
print(token_ids.shape)

print("\nEmbedding output shape:")
print(output.shape)