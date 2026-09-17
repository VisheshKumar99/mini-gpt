import time
import torch
import torch.nn as nn

from src.tokenizer import CharacterTokenizer
from src.dataset import TextDataset
from src.gpt import GPT
from backend.app import database


class ExperimentRunner:

    def __init__(self, config):
        self.config = config

        self.device = self._get_device()

        self.metrics = []

    def _get_device(self):
        if self.config.get("device") == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")

        if self.config.get("device") == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")

        return torch.device("cpu")

    def run(self):

        print("================================")
        print("Starting experiment")
        print("================================")

        print("Dataset:", self.config["dataset_file"])
        print("Model:", self.config["model_id"])
        print("Batch size:", self.config["batch_size"])
        print("Learning rate:", self.config["learning_rate"])
        print("Steps:", self.config["max_steps"])
        print("Device:", self.device)

        # ------------------------------------------------
        # Load dataset
        # ------------------------------------------------

        with open(
            self.config["dataset_file"],
            "r",
            encoding="utf-8"
        ) as f:
            text = f.read()

        print("Dataset characters:", len(text))

        # ------------------------------------------------
        # Tokenizer
        # ------------------------------------------------

        tokenizer = CharacterTokenizer(text)

        vocab_size = tokenizer.vocab_size

        print("Vocabulary:", vocab_size)

        # ------------------------------------------------
        # Dataset
        # ------------------------------------------------

        dataset = TextDataset(
            text=text,
            tokenizer=tokenizer,
            context_length=self.config["block_size"]
        )

        # ------------------------------------------------
        # Model
        # ------------------------------------------------

        model = GPT(
            vocab_size=vocab_size,
            embedding_dim=self.config["n_embd"],
            num_heads=self.config["n_head"],
            num_layers=self.config["n_layer"],
            context_length=self.config["block_size"]
        ).to(self.device)

        actual_params = sum(
            p.numel()
            for p in model.parameters()
        )

        print("Actual parameters:", actual_params)

        # ------------------------------------------------
        # Training
        # ------------------------------------------------

        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["learning_rate"]
        )

        model.train()

        start_time = time.time()

        for step in range(1, self.config["max_steps"] + 1):

            x, y = dataset.get_batch(
                self.config["batch_size"]
            )

            x = x.to(self.device)
            y = y.to(self.device)

            logits, _ = model(x)

            loss = loss_fn(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            elapsed = time.time() - start_time

            tokens_seen = (
                step
                * self.config["batch_size"]
                * self.config["block_size"]
            )

            tokens_per_sec = (
                tokens_seen / elapsed
                if elapsed > 0
                else 0
            )

            metric = {
                "step": step,
                "train_loss": loss.item(),
                "val_loss": None,
                "tokens_per_sec": tokens_per_sec,
                "tokens_seen": tokens_seen,
                "elapsed_s": elapsed,
            }

            

            self.metrics.append(metric)
            
            database.add_experiment_metric(
                self.config["experiment_id"],
                metric
            )

            # Don't print every step
            if step % 100 == 0 or step == 1:

                print(
                    f"Step {step:5d} | "
                    f"Loss {loss.item():.4f} | "
                    f"TPS {tokens_per_sec:.0f}"
                )

        total_time = time.time() - start_time

        print()
        print("================================")
        print("Training completed")
        print("================================")

        return {
            "status": "completed",
            "actual_parameters": actual_params,
            "vocab_size": vocab_size,
            "training_time": total_time,
            "metrics": self.metrics,
            "model": model,
        }