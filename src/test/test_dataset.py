from src.tokenizer import CharacterTokenizer
from src.dataset import TextDataset


with open("data/input_10MB.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer = CharacterTokenizer(text)

dataset = TextDataset(
    text=text,
    tokenizer=tokenizer,
    context_length=8
)


x, y = dataset.get_batch(batch_size=2)


print("Vocabulary size:")
print(tokenizer.vocab_size)

print("\nX shape:")
print(x.shape)

print("\nY shape:")
print(y.shape)

print("\nX:")
print(x)

print("\nY:")
print(y)

print("\nDecoded X:")
print(tokenizer.decode(x[0].tolist()))

print("\nDecoded Y:")
print(tokenizer.decode(y[0].tolist()))