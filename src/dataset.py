import torch


class TextDataset:
    def __init__(self, text, tokenizer, context_length):

        self.tokenizer = tokenizer
        self.context_length = context_length

        # Convert entire text into token IDs
        self.tokens = torch.tensor(
            tokenizer.encode(text),
            dtype=torch.long
        )

    def get_batch(self, batch_size):

        # Random starting positions
        starts = torch.randint(
            0,
            len(self.tokens) - self.context_length - 1,
            (batch_size,)
        )

        x = torch.stack([
            self.tokens[i:i + self.context_length]
            for i in starts
        ])

        y = torch.stack([
            self.tokens[i + 1:i + self.context_length + 1]
            for i in starts
        ])

        return x, y