import torch
import torch.nn as nn

from src.tokenizer import CharacterTokenizer
from src.dataset import TextDataset
from src.gpt import GPT


# --------------------------------
# Configuration
# --------------------------------

CONTEXT_LENGTH = 128
EMBEDDING_DIM = 16
NUM_HEADS = 4
NUM_LAYERS = 2
BATCH_SIZE = 4
LEARNING_RATE = 3e-4


# --------------------------------
# Load text
# --------------------------------

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# --------------------------------
# Tokenizer
# --------------------------------

tokenizer = CharacterTokenizer(text)

vocab_size = tokenizer.vocab_size

print("Vocabulary size:", vocab_size)


# --------------------------------
# Dataset
# --------------------------------

dataset = TextDataset(
    text=text,
    tokenizer=tokenizer,
    context_length=CONTEXT_LENGTH
)


# --------------------------------
# Model
# --------------------------------

model = GPT(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    context_length=CONTEXT_LENGTH
)


# --------------------------------
# Loss + Optimizer
# --------------------------------

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------
# Training step
# --------------------------------

x, y = dataset.get_batch(BATCH_SIZE)

# Forward
logits, _ = model(x)

# Loss
loss = loss_fn(
    logits.reshape(-1, vocab_size),
    y.reshape(-1)
)

print("\nLoss BEFORE backward:")
print(loss.item())


# --------------------------------
# Backpropagation
# --------------------------------

optimizer.zero_grad()

loss.backward()


# --------------------------------
# Check gradients
# --------------------------------

print("\nGradient example:")

for name, parameter in model.named_parameters():

    if parameter.grad is not None:
        print(name)
        print("Gradient shape:", parameter.grad.shape)
        print("Gradient mean:", parameter.grad.mean().item())
        break


# --------------------------------
# Update weights
# --------------------------------

optimizer.step()

print("\nOptimizer step completed.")