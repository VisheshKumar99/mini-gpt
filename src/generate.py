import torch

from src.tokenizer import CharacterTokenizer
from src.gpt import GPT


# -------------------------
# Configuration
# -------------------------

CONTEXT_LENGTH = 128
MAX_NEW_TOKENS = 100


# -------------------------
# Device
# -------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# -------------------------
# Load dataset
# -------------------------

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# -------------------------
# Tokenizer
# -------------------------

tokenizer = CharacterTokenizer(text)

vocab_size = tokenizer.vocab_size


# -------------------------
# Create model
# -------------------------

model = GPT(
    vocab_size=vocab_size,
    embedding_dim=16,
    num_heads=4,
    num_layers=2,
    context_length=CONTEXT_LENGTH
).to(device)


# -------------------------
# Load trained weights
# -------------------------

checkpoint = torch.load(
    "model.pt",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# -------------------------
# Starting prompt
# -------------------------

prompt = "The"

context = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device
)


# -------------------------
# Generate
# -------------------------

with torch.no_grad():

    for _ in range(MAX_NEW_TOKENS):

        # Keep only the last context window
        input_context = context[:, -CONTEXT_LENGTH:]

        # Forward pass
        logits, _ = model(input_context)

        # Last token's prediction
        next_token_logits = logits[:, -1, :]

        # Convert logits -> probabilities
        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        # Sample next token
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Append token
        context = torch.cat(
            [context, next_token],
            dim=1
        )


# -------------------------
# Decode
# -------------------------

generated_text = tokenizer.decode(
    context[0].tolist()
)

print("\nGenerated text:")
print(generated_text)