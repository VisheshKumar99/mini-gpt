import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim, context_length):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.context_length = context_length

        # X -> Query
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # X -> Key
        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # X -> Value
        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # Causal mask
        #
        # Example for context_length = 4:
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        #
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_length,
                    context_length
                )
            )
        )

    def forward(self, x):

        # x shape:
        #
        # [batch_size, sequence_length, embedding_dim]
        #
        # Example:
        # [32, 128, 384]

        batch_size, sequence_length, embedding_dim = x.shape

        # -----------------------------------------
        # 1. Create Query, Key and Value
        # -----------------------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Shapes:
        #
        # Q = [B, T, C]
        # K = [B, T, C]
        # V = [B, T, C]

        # -----------------------------------------
        # 2. Calculate attention scores
        # -----------------------------------------

        scores = Q @ K.transpose(-2, -1)

        # Shape:
        #
        # [B, T, C]
        #      @
        # [B, C, T]
        #
        # =
        #
        # [B, T, T]

        # -----------------------------------------
        # 3. Scale scores
        # -----------------------------------------

        scores = scores / math.sqrt(embedding_dim)

        # -----------------------------------------
        # 4. Apply causal mask
        # -----------------------------------------

        scores = scores.masked_fill(
            self.mask[:sequence_length, :sequence_length] == 0,
            float("-inf")
        )

        # -----------------------------------------
        # 5. Softmax
        # -----------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # -----------------------------------------
        # 6. Weighted sum of Values
        # -----------------------------------------

        output = attention_weights @ V

        # output:
        #
        # [B, T, T]
        #      @
        # [B, T, C]
        #
        # =
        #
        # [B, T, C]

        return output, attention_weights