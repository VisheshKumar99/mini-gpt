class CharacterTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))

        self.stoi = {
            ch: i
            for i, ch in enumerate(self.chars)
        }

        self.itos = {
            i: ch
            for i, ch in enumerate(self.chars)
        }

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text: str):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return "".join(self.itos[token] for token in tokens)

token = CharacterTokenizer("Hello, world!")
# print(token.chars)
# print(token.stoi)
# print(token.itos)
# print(token.encode("Hello, world!"))
# print(token.decode(token.encode("Hello, world!")))
# print(token.vocab_size)

