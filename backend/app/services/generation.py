"""Text generation from a saved per-experiment GPT checkpoint.

Checkpoints are written by ExperimentRunner.run() to models/experiment_{id}.pt
and contain the model weights, architecture hyperparameters, and the character
tokenizer vocabulary. This module loads one and samples a continuation from a
prompt, mirroring the logic in src/generate.py.
"""

from __future__ import annotations

import torch

from src.gpt import GPT
from backend.app.services.experiment_runner import checkpoint_path


class ModelNotTrainedError(Exception):
    """Raised when no saved checkpoint exists for an experiment."""


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Cache loaded models so each chat turn does not re-read the checkpoint.
# Keyed by experiment id -> (model, chars, context_length, device).
_CACHE: dict[int, tuple] = {}


def _load(experiment_id: int):
    if experiment_id in _CACHE:
        return _CACHE[experiment_id]

    path = checkpoint_path(experiment_id)
    if not path.exists():
        raise ModelNotTrainedError(
            f"No trained model for experiment {experiment_id}. Train it first."
        )

    device = _get_device()
    checkpoint = torch.load(path, map_location=device)

    model = GPT(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        context_length=checkpoint["context_length"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    chars = checkpoint["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    context_length = checkpoint["context_length"]

    entry = (model, stoi, itos, context_length, device)
    _CACHE[experiment_id] = entry
    return entry


def has_checkpoint(experiment_id: int) -> bool:
    """Whether a trained model exists on disk for this experiment."""
    return checkpoint_path(experiment_id).exists()


def generate_text(
    experiment_id: int,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 20,
) -> str:
    """Sample a continuation of ``prompt`` from the experiment's model.

    Returns only the newly generated text (not the prompt echoed back).
    Characters in the prompt that the model never saw during training are
    skipped, since the character tokenizer has no id for them.
    """
    model, stoi, itos, context_length, device = _load(experiment_id)

    # Encode prompt, dropping unknown characters. Fall back to a space so the
    # model always has at least one starting token.
    encoded = [stoi[ch] for ch in prompt if ch in stoi]
    if not encoded:
        encoded = [stoi.get(" ", 0)]

    context = torch.tensor([encoded], dtype=torch.long, device=device)
    start_len = context.size(1)

    temperature = max(float(temperature), 1e-6)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_context = context[:, -context_length:]
            logits, _ = model(input_context)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, next_token_logits.size(-1))
                values, indices = torch.topk(next_token_logits, k)
                filtered = torch.full_like(next_token_logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                next_token_logits = filtered

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)

    generated_ids = context[0, start_len:].tolist()
    return "".join(itos[i] for i in generated_ids)
