import torch


class TextDataset:
    def __init__(
        self,
        text: str,
        tokenizer,
        context_length: int,
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length

        tokens = tokenizer.encode(text)

        split_index = int(len(tokens) * train_ratio)

        if split == "train":
            self.data = torch.tensor(
                tokens[:split_index],
                dtype=torch.long,
            )
        elif split == "val":
            self.data = torch.tensor(
                tokens[split_index:],
                dtype=torch.long,
            )
        else:
            raise ValueError(
                f"Unknown split: {split}. Use 'train' or 'val'."
            )

        if len(self.data) <= context_length:
            raise ValueError(
                f"{split} dataset is too small. "
                f"Need more than {context_length} tokens."
            )

    def get_batch(self, batch_size: int):
        max_start = len(self.data) - self.context_length

        starts = torch.randint(
            0,
            max_start,
            (batch_size,),
        )

        x = torch.stack([
            self.data[start:start + self.context_length]
            for start in starts
        ])

        y = torch.stack([
            self.data[start + 1:start + self.context_length + 1]
            for start in starts
        ])

        return x, y