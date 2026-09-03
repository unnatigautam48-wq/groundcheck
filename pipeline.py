from dataclasses import dataclass
from typing import List

from retriever import Chunk, Retriever
from verifier import ClaimCheck, groundedness_score, verify_answer


@dataclass
class RAGResult:
    query: str
    answer: str
    claim_checks: List[ClaimCheck]
    score: float


class RAGPipeline:
    def __init__(self, chunks: List[Chunk]):
        self.retriever = Retriever(chunks)

    def _generate_answer(self, query: str, evidence_texts: List[str], top_n: int = 2) -> str:
        """Extractive generation: pick the sentences most relevant to the query
        from the retrieved evidence and join them into an answer."""
        from verifier import split_into_claims
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from textutils import tokenize, STOPWORDS

        sentences = []
        for text in evidence_texts:
            sentences.extend(split_into_claims(text))
        if not sentences:
            return "No relevant information found."

        vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, stop_words=STOPWORDS)
        matrix = vectorizer.fit_transform([query] + sentences)
        sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
        ranked = sorted(zip(sentences, sims), key=lambda x: x[1], reverse=True)
        best = [s for s, _ in ranked[:top_n]]
        return " ".join(best)

    def ask(self, query: str, k: int = 3, injected_answer: str = None) -> RAGResult:
        retrieved = self.retriever.retrieve(query, k=k)
        evidence_texts = [r.chunk.text for r in retrieved]

        answer = injected_answer if injected_answer else self._generate_answer(query, evidence_texts)
        checks = verify_answer(answer, evidence_texts)
        score = groundedness_score(checks)

        return RAGResult(query=query, answer=answer, claim_checks=checks, score=score)
