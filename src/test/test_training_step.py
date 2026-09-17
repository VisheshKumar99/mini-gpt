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

print("Vocabulary size:")
print(vocab_size)


# --------------------------------
# Dataset
# --------------------------------

dataset = TextDataset(
    text=text,
    tokenizer=tokenizer,
    context_length=CONTEXT_LENGTH
)


x, y = dataset.get_batch(BATCH_SIZE)

print("\nInput X:")
print(x.shape)

print("\nTarget Y:")
print(y.shape)


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
# Forward pass
# --------------------------------

logits, _ = model(x)

print("\nLogits:")
print(logits.shape)


# --------------------------------
# Calculate loss
# --------------------------------

loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(
    logits.reshape(-1, vocab_size),
    y.reshape(-1)
)


print("\n================================")
print("TRAINING STEP")
print("================================")

print("\nLogits flattened:")
print(logits.reshape(-1, vocab_size).shape)

print("\nTargets flattened:")
print(y.reshape(-1).shape)

print("\nLoss:")
print(loss.item())