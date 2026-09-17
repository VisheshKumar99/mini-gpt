import torch
import torch.nn as nn

from src.embedding import GPTEmbedding
from src.transformer import TransformerBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads,
        num_layers,
        context_length
    ):
        super().__init__()

        self.embedding = GPTEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_length=context_length
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                context_length=context_length
            )
            for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        # Convert hidden representation -> vocabulary logits
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

    def forward(self, token_ids, verbose=False):

        # -----------------------------
        # 1. Token + Position Embedding
        # -----------------------------

        x = self.embedding(token_ids)
        if verbose:
            print("\n==============================")
            print("GPT Forward Pass")
            print("==============================")

            print("\nAfter Embedding:")
            print(x.shape)

        # -----------------------------
        # 2. Transformer Blocks
        # -----------------------------

        attention_weights = []

        for i, block in enumerate(self.transformer_blocks):
            if verbose:
                print(f"\n========== Block {i + 1} ==========")

            x, weights = block(x)

            attention_weights.append(weights)

        # -----------------------------
        # 3. Final LayerNorm
        # -----------------------------

        x = self.final_layer_norm(x)

        if verbose:
            print("\nAfter Final LayerNorm:")
            print(x.shape)

        # -----------------------------
        # 4. Language Model Head
        # -----------------------------

        logits = self.lm_head(x)

        if verbose:
            print("\nFinal Logits:")
            print(logits.shape)

        return logits, attention_weights