from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from textutils import tokenize, STOPWORDS


@dataclass
class Chunk:
    doc_id: str
    text: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("Retriever needs at least one chunk to index.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None,
            stop_words=STOPWORDS, ngram_range=(1, 2),
        )
        self._matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedChunk]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]))
            for i in top_idx
            if scores[i] > 0
        ]
