import torch
import torch.nn as nn

from src.attention_multi import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length
    ):
        super().__init__()

        # Layer Normalization
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        # Multi-Head Self Attention
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length
        )

        # Feed Forward Network
        self.ffn = FeedForward(embedding_dim)

    def forward(self, x, verbose=False):

        if verbose:
            print("\n==============================")
            print("Transformer Block")
            print("==============================")

        if verbose:
            print("\nInput:")
            print(x.shape)

        # --------------------------------
        # 1. Attention + Residual
        # --------------------------------

        normalized_x = self.ln1(x)

        if verbose:
            print("\nAfter LayerNorm 1:")
            print(normalized_x.shape)

        attention_output, attention_weights = self.attention(normalized_x)

        if verbose:
            print("\nAttention output:")
            print(attention_output.shape)

        # Residual connection
        x = x + attention_output

        if verbose:
            print("\nAfter Attention Residual:")
            print(x.shape)

        # --------------------------------
        # 2. Feed Forward + Residual
        # --------------------------------

        normalized_x = self.ln2(x)

        if verbose:
            print("\nAfter LayerNorm 2:")
            print(normalized_x.shape)

        ffn_output = self.ffn(normalized_x)

        if verbose:
            print("\nFFN output:")
            print(ffn_output.shape)

        # Residual connection
        x = x + ffn_output
        if verbose:
            print("\nAfter FFN Residual:")
            print(x.shape)

        return x, attention_weights