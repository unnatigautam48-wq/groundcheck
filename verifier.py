import re
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from textutils import tokenize, STOPWORDS

SUPPORTED_THRESHOLD = 0.28
WEAK_THRESHOLD = 0.12

NEGATION_WORDS = {"not", "no", "never", "cannot", "can't", "won't", "isn't",
                   "aren't", "doesn't", "don't", "excluding", "except"}


@dataclass
class ClaimCheck:
    claim: str
    best_match_score: float
    best_match_source: str
    label: str  # SUPPORTED | WEAK | UNSUPPORTED
    reason: str = ""  # why it was flagged, when relevant


def split_into_claims(answer: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def label_for_score(score: float) -> str:
    if score >= SUPPORTED_THRESHOLD:
        return "SUPPORTED"
    if score >= WEAK_THRESHOLD:
        return "WEAK"
    return "UNSUPPORTED"


def extract_numbers(text: str) -> set:
    """Pulls out anything number-like: 500GB, $9, 6 PM, 30 days, 24/7, etc.
    Two claims that are otherwise near-identical but disagree on these are
    almost always a hallucinated fact, so this catches what pure cosine
    similarity misses (word overlap stays high even when the number is wrong)."""
    return set(re.findall(r"\d+(?:\.\d+)?%?", text.lower()))


def has_negation(text: str) -> bool:
    words = set(re.findall(r"[a-z']+", text.lower()))
    return bool(words & NEGATION_WORDS)


def fact_conflict(claim: str, evidence: str) -> str:
    """Returns a short reason string if claim and its best-matching evidence
    disagree on numbers or negation, else empty string."""
    claim_nums, evidence_nums = extract_numbers(claim), extract_numbers(evidence)
    unsupported_nums = claim_nums - evidence_nums
    if unsupported_nums and evidence_nums:
        return (f"claim states {', '.join(sorted(unsupported_nums))} — not found "
                f"in matched evidence ({', '.join(sorted(evidence_nums)) or 'none'})")
    if has_negation(claim) != has_negation(evidence):
        return "negation mismatch (claim and evidence disagree on yes/no)"
    return ""


def verify_answer(answer: str, evidence_texts: List[str]) -> List[ClaimCheck]:
    """
    evidence_texts: the raw text of the chunks the retriever pulled back.
    Each claim in the answer is scored against every evidence sentence,
    and the strongest match decides the label.
    """
    claims = split_into_claims(answer)
    if not claims:
        return []

    evidence_sentences = []
    for text in evidence_texts:
        evidence_sentences.extend(split_into_claims(text))
    if not evidence_sentences:
        return [ClaimCheck(c, 0.0, "", "UNSUPPORTED") for c in claims]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, token_pattern=None,
        stop_words=STOPWORDS, ngram_range=(1, 2),
    )
    all_text = claims + evidence_sentences
    matrix = vectorizer.fit_transform(all_text)

    claim_vecs = matrix[: len(claims)]
    evidence_vecs = matrix[len(claims):]
    sim_matrix = cosine_similarity(claim_vecs, evidence_vecs)

    results = []
    for i, claim in enumerate(claims):
        best_idx = int(sim_matrix[i].argmax())
        best_score = float(sim_matrix[i][best_idx])
        best_source = evidence_sentences[best_idx]

        label = label_for_score(best_score)
        reason = ""
        conflict = fact_conflict(claim, best_source)
        if conflict and label != "UNSUPPORTED":
            label = "UNSUPPORTED"
            reason = conflict

        results.append(
            ClaimCheck(
                claim=claim,
                best_match_score=round(best_score, 3),
                best_match_source=best_source,
                label=label,
                reason=reason,
            )
        )
    return results


def groundedness_score(checks: List[ClaimCheck]) -> float:
    if not checks:
        return 0.0
    weights = {"SUPPORTED": 1.0, "WEAK": 0.5, "UNSUPPORTED": 0.0}
    return round(sum(weights[c.label] for c in checks) / len(checks), 3)
