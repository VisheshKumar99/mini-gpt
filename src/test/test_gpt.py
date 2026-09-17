import torch

from src.gpt import GPT


BATCH_SIZE = 2
CONTEXT_LENGTH = 8
VOCAB_SIZE = 100
EMBEDDING_DIM = 16
NUM_HEADS = 4
NUM_LAYERS = 2


token_ids = torch.randint(
    0,
    VOCAB_SIZE,
    (BATCH_SIZE, CONTEXT_LENGTH)
)


model = GPT(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    context_length=CONTEXT_LENGTH
)


logits, attention_weights = model(token_ids)


print("\n================================")
print("FINAL RESULTS")
print("================================")

print("\nToken IDs:")
print(token_ids.shape)

print("\nLogits:")
print(logits.shape)

print("\nAttention blocks:")
print(len(attention_weights))

print("\nAttention shape from block 1:")
print(attention_weights[0].shape)

print("\nAttention shape from block 2:")
print(attention_weights[1].shape)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nTotal parameters:")
print(f"{total_params:,}")

print("\nTrainable parameters:")
print(f"{trainable_params:,}")