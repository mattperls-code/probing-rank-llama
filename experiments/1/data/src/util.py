import string

def tokenize(text: str) -> list[str]:
    return [term.lower() for term in text.translate(str.maketrans('', '', string.punctuation)).split() if term]