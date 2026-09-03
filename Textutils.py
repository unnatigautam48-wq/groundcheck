import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def light_stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def tokenize(text: str):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [light_stem(w) for w in words]



STOPWORDS = sorted(set(light_stem(w) for w in ENGLISH_STOP_WORDS))
