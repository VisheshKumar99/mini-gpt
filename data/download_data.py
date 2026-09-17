from datasets import load_dataset

TARGET_MB = 1
TARGET_BYTES = TARGET_MB * 1024 * 1024

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True
)

output_file = "data/input_1MB.txt"

total_bytes = 0
stories = 0

with open(output_file, "w", encoding="utf-8") as f:

    for row in dataset:
        story = row["text"]

        # +1 gives each story a newline
        story_bytes = len((story + "\n").encode("utf-8"))

        f.write(story + "\n")

        total_bytes += story_bytes
        stories += 1

        if total_bytes >= TARGET_BYTES:
            break

print(f"Stories: {stories}")
print(f"Size: {total_bytes / (1024 * 1024):.2f} MB")
print(f"Saved to: {output_file}")