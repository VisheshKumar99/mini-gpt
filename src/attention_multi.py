import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length
    ):
        super().__init__()

        # --------------------------------
        # Configuration
        # --------------------------------

        assert embedding_dim % num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.context_length = context_length

        # --------------------------------
        # Q, K, V projections
        # --------------------------------

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # --------------------------------
        # Output projection
        # --------------------------------

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # --------------------------------
        # Causal mask
        # --------------------------------

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_length,
                    context_length
                )
            )
        )

    def forward(self, x, verbose=False):

        # ==========================================
        # Input
        # ==========================================

        batch_size, sequence_length, embedding_dim = x.shape

        if verbose:
            print("\n--- Multi Head Attention ---")

            print("Input X:")
            print(x.shape)

        # ==========================================
        # 1. Create Q, K, V
        # ==========================================

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        if verbose:
            print("\nAfter Q projection:")
            print(Q.shape)

            print("After K projection:")
            print(K.shape)

            print("After V projection:")
            print(V.shape)

        # ==========================================
        # 2. Split into multiple heads
        # ==========================================

        #
        # Current:
        #
        # [B, T, C]
        #
        # Example:
        #
        # [2, 8, 16]
        #
        # becomes:
        #
        # [B, T, num_heads, head_dim]
        #
        # [2, 8, 4, 4]
        #

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        if verbose:
            print("\nAfter splitting heads:")
            print("Q:", Q.shape)
            print("K:", K.shape)
            print("V:", V.shape)

        # ==========================================
        # 3. Move heads before sequence dimension
        # ==========================================

        #
        # [B, T, H, D]
        #
        # becomes:
        #
        # [B, H, T, D]
        #

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        if verbose:
            print("\nAfter transpose:")
            print("Q:", Q.shape)
            print("K:", K.shape)
            print("V:", V.shape)

        # ==========================================
        # 4. Calculate attention scores
        # ==========================================

        #
        # Q:
        # [B, H, T, D]
        #
        # K:
        # [B, H, T, D]
        #
        # K.transpose:
        # [B, H, D, T]
        #
        # Result:
        # [B, H, T, T]
        #

        scores = Q @ K.transpose(-2, -1)

        if verbose:
            print("\nAttention scores:")
            print(scores.shape)

        # ==========================================
        # 5. Scale
        # ==========================================

        scores = scores / math.sqrt(self.head_dim)

        if verbose:
            print("\nAfter scaling:")
            print(scores.shape)

        # ==========================================
        # 6. Causal mask
        # ==========================================

        scores = scores.masked_fill(
            self.mask[:sequence_length, :sequence_length] == 0,
            float("-inf")
        )

        if verbose:
            print("\nAfter causal mask:")
            print(scores.shape)

        # ==========================================
        # 7. Softmax
        # ==========================================

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        if verbose:
            print("\nAttention weights:")
            print(attention_weights.shape)

        # ==========================================
        # 8. Weighted sum of V
        # ==========================================

        output = attention_weights @ V

        if verbose:
            print("\nAttention output per head:")
            print(output.shape)

        # ==========================================
        # 9. Combine heads
        # ==========================================

        #
        # Current:
        #
        # [B, H, T, D]
        #
        # Move T before H:
        #
        # [B, T, H, D]
        #

        output = output.transpose(1, 2)

        if verbose:
            print("\nAfter moving heads:")
            print(output.shape)

        # ==========================================
        # 10. Concatenate heads
        # ==========================================

        #
        # [B, T, H, D]
        #
        # becomes:
        #
        # [B, T, H * D]
        #
        # H * D = C
        #

        output = output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        if verbose:
            print("\nAfter concatenating heads:")
            print(output.shape)

        # ==========================================
        # 11. Output projection
        # ==========================================

        output = self.output_projection(output)

        if verbose:
            print("\nFinal attention output:")
            print(output.shape)

        return output, attention_weights