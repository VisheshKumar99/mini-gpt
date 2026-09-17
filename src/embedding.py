import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        context_length
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

        self.context_length = context_length

    def forward(self, token_ids):

        batch_size, sequence_length = token_ids.shape

        # Token embeddings
        token_embeddings = self.token_embedding(token_ids)

        # Position IDs: 0, 1, 2, ..., sequence_length - 1
        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        # Position embeddings
        position_embeddings = self.position_embedding(positions)

        # Combine token + position information
        embeddings = token_embeddings + position_embeddings

        return embeddings