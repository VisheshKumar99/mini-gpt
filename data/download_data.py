"""Download text from TinyStories into a plain .txt file for training.

Usage:
    python data/download_data.py                 # default: 100 MB -> data/100mb.txt
    python data/download_data.py 10              # 10 MB -> data/10mb.txt
    python data/download_data.py 100 data/big.txt
"""

import sys

from datasets import load_dataset


def main() -> None:
    target_mb = float(sys.argv[1]) if len(sys.argv) > 1 else 100
    # Default output name matches the app's dataset catalog (e.g. data/100mb.txt).
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"data/{int(target_mb)}mb.txt"

    target_bytes = int(target_mb * 1024 * 1024)

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )

    total_bytes = 0
    stories = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset:
            line = row["text"] + "\n"
            f.write(line)

            total_bytes += len(line.encode("utf-8"))
            stories += 1

            if stories % 5000 == 0:
                print(f"  {total_bytes / (1024 * 1024):.1f} MB…", flush=True)

            if total_bytes >= target_bytes:
                break

    print(f"Stories: {stories}")
    print(f"Size: {total_bytes / (1024 * 1024):.2f} MB")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
