import torch
from pathlib import Path

from src.tokenizer import CharacterTokenizer
from src.gpt import GPT


# ==========================================
# Configuration
# ==========================================

CONTEXT_LENGTH = 128
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 10

PROMPT = "The"


# ==========================================
# Project path
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model.pt"
DATA_PATH = PROJECT_ROOT / "data" / "input.txt"


# ==========================================
# Device
# ==========================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


# ==========================================
# Load dataset
# ==========================================

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()


# ==========================================
# Tokenizer
# ==========================================

tokenizer = CharacterTokenizer(text)
vocab_size = tokenizer.vocab_size


# ==========================================
# Load checkpoint
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)


# ==========================================
# Create model
# ==========================================

model = GPT(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    context_length=checkpoint["context_length"],
).to(device)


# ==========================================
# Load trained weights
# ==========================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ==========================================
# Encode prompt
# ==========================================

context = torch.tensor(
    [tokenizer.encode(PROMPT)],
    dtype=torch.long,
    device=device,
)


# ==========================================
# Generate
# ==========================================

with torch.no_grad():

    for _ in range(MAX_NEW_TOKENS):

        # Keep only the model's context window
        input_context = context[:, -CONTEXT_LENGTH:]

        # Forward pass
        logits, _ = model(input_context)

        # Get prediction for the last token
        next_token_logits = logits[:, -1, :]

        # --------------------------------------
        # Temperature
        # --------------------------------------

        next_token_logits = next_token_logits / TEMPERATURE

        # --------------------------------------
        # Top-K
        # --------------------------------------

        if TOP_K is not None:

            values, indices = torch.topk(
                next_token_logits,
                min(
                    TOP_K,
                    next_token_logits.size(-1),
                ),
            )

            filtered_logits = torch.full_like(
                next_token_logits,
                float("-inf"),
            )

            filtered_logits.scatter_(
                1,
                indices,
                values,
            )

            next_token_logits = filtered_logits

        # --------------------------------------
        # Convert logits to probabilities
        # --------------------------------------

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        # --------------------------------------
        # Sample next token
        # --------------------------------------

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        # --------------------------------------
        # Append token
        # --------------------------------------

        context = torch.cat(
            [
                context,
                next_token,
            ],
            dim=1,
        )


# ==========================================
# Decode
# ==========================================

generated_text = tokenizer.decode(
    context[0].tolist()
)


# ==========================================
# Output
# ==========================================

print()
print("Prompt:", PROMPT)
print("Temperature:", TEMPERATURE)
print("Top-K:", TOP_K)

print()
print("Generated text:")
print(generated_text)