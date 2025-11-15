import re
from collections import Counter
from typing import List


_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","of","to","in","on","at","by","with",
    "is","are","was","were","be","been","being","as","it","this","that","these","those","from"
}


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter using punctuation and newlines."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _tokenize(text: str) -> List[str]:
    """Tokenize text into words, lowercased, alphanumeric only."""
    return [w for w in re.findall(r"[A-Za-z0-9']+", text.lower()) if w]


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Extractive summarization: rank sentences by term frequency and select top ones.

    - Keeps original order of selected sentences.
    - Falls back gracefully for short inputs.
    """
    text = (text or "").strip()
    if not text:
        return "Summary:"

    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return f"Summary: {' '.join(sentences)}"

    words = [w for w in _tokenize(text) if w not in _STOPWORDS]
    if not words:
        # If everything is a stopword, just take first sentences
        return f"Summary: {' '.join(sentences[:max_sentences])}"

    freq = Counter(words)

    # Score sentences by sum of word frequencies
    scored = []
    for idx, s in enumerate(sentences):
        sw = _tokenize(s)
        score = sum(freq[w] for w in sw if w not in _STOPWORDS)
        scored.append((idx, score))

    # Pick top N by score
    top_idxs = [idx for idx, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:max_sentences]]
    top_idxs.sort()  # restore original order
    selected = [sentences[i] for i in top_idxs]
    return f"Summary: {' '.join(selected)}"


def compute_cognitive_score(summary: str) -> float:
    """Heuristic cognitive score in [0,1].

    Factors:
    - lexical diversity (unique/total words)
    - structural richness (sentences normalized)
    - length normalization (longer summaries up to a cap)
    """
    s = (summary or "").strip()
    if not s:
        return 0.0

    words = _tokenize(s)
    total = len(words)
    unique = len(set(words))
    diversity = (unique / total) if total > 0 else 0.0

    sentences = _split_sentences(s)
    structure = min(len(sentences) / 3.0, 1.0)

    length_norm = min(len(s) / 300.0, 1.0)

    score = 0.4 * diversity + 0.4 * structure + 0.2 * length_norm
    # clamp to [0,1]
    return max(0.0, min(score, 1.0))