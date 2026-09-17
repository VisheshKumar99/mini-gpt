import torch
import torch.nn as nn

from src.tokenizer import CharacterTokenizer
from src.dataset import TextDataset
from src.gpt import GPT


# ==========================================
# Configuration
# ==========================================

CONTEXT_LENGTH = 128
EMBEDDING_DIM = 16
NUM_HEADS = 4
NUM_LAYERS = 2

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
TRAINING_STEPS = 1000
MODEL_PATH = "model.pt"


# ==========================================
# Load dataset
# ==========================================

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Dataset size:", len(text), "characters")


# ==========================================
# Tokenizer
# ==========================================

tokenizer = CharacterTokenizer(text)

vocab_size = tokenizer.vocab_size

print("Vocabulary size:", vocab_size)


# ==========================================
# Dataset
# ==========================================

dataset = TextDataset(
    text=text,
    tokenizer=tokenizer,
    context_length=CONTEXT_LENGTH
)


# ==========================================
# Device
# ==========================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


# ==========================================
# Model
# ==========================================

model = GPT(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    context_length=CONTEXT_LENGTH
)

model = model.to(device)


# ==========================================
# Loss
# ==========================================

loss_fn = nn.CrossEntropyLoss()


# ==========================================
# Optimizer
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# Parameter count
# ==========================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("Total parameters:", f"{total_params:,}")


# ==========================================
# Training
# ==========================================

model.train()

for step in range(TRAINING_STEPS):

    # Get batch
    x, y = dataset.get_batch(BATCH_SIZE)

    x = x.to(device)
    y = y.to(device)

    # Forward
    logits, _ = model(x)

    # Loss
    loss = loss_fn(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    # Clear gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    # Logging
    if step % 100 == 0:
        print(
            f"Step {step:4d} | Loss {loss.item():.4f}"
        )

checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "embedding_dim": EMBEDDING_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "context_length": CONTEXT_LENGTH,
    "chars": tokenizer.chars,
}

torch.save(checkpoint, MODEL_PATH)

print()
print("Training completed.")
print("Model saved to:", MODEL_PATH)