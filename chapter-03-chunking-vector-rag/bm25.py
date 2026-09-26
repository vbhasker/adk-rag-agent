"""BM25 (Okapi BM25) implemented from scratch, plus TF-IDF for comparison.

No libraries on purpose: every number in the ranking is computed right here, so you can
print it, question it and tweak it.

BM25 score of document D for query Q = sum over query terms t of:

    IDF(t) * tf(t, D) * (k1 + 1)
             -----------------------------------------------
             tf(t, D) + k1 * (1 - b + b * |D| / avgdl)

    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))     # Lucene / Elasticsearch variant

    tf     = how often t appears in D
    df     = how many documents contain t
    |D|    = document length in tokens, avgdl = average document length
    k1     = term-frequency saturation (1.2 - 2.0 typical). Higher = repeated words keep counting.
    b      = length normalisation (0 = ignore length, 1 = fully normalise). 0.75 is the classic default.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# A deliberately small stop-word list. Real engines ship language-specific lists.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from",
    "how", "i", "if", "in", "is", "it", "its", "my", "of", "on", "or", "our", "so", "that", "the",
    "this", "to", "was", "what", "when", "which", "with", "you", "your", "will", "we", "not",
}

# Keeps codes such as "e-07", "4.2.1", "un3480" and "700x40c" as single tokens.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def light_stem(token: str) -> str:
    """A toy stemmer: 'charge', 'charges', 'charging' and 'charged' all become 'charg'.

    Production systems use Porter/Snowball stemmers or lemmatisers. The point here is to show
    that those words only match each other if you normalise them somehow.
    """
    if len(token) <= 3 or any(ch.isdigit() for ch in token) or token.endswith(("ss", "eed", "us")):
        return token
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""), ("e", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + replacement
    return token


def tokenize(text: str, remove_stop_words: bool = True, stem: bool = True) -> list[str]:
    text = text.lower().replace("’", "'").replace("won't", "will not").replace("can't", "cannot")
    tokens = TOKEN_PATTERN.findall(text.replace("n't", " not"))
    if remove_stop_words:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if stem:
        tokens = [light_stem(t) for t in tokens]
    return tokens


@dataclass
class ScoredDoc:
    index: int
    score: float
    # term -> contribution to the score, so we can *explain* the ranking
    breakdown: dict[str, float] = field(default_factory=dict)


class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tf = [Counter(tokens) for tokens in corpus_tokens]
        self.doc_len = [len(tokens) for tokens in corpus_tokens]
        self.n_docs = len(corpus_tokens)
        self.avgdl = sum(self.doc_len) / self.n_docs
        self.df: Counter[str] = Counter()
        for tf in self.doc_tf:
            self.df.update(tf.keys())

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score_doc(self, query_terms: list[str], i: int) -> ScoredDoc:
        tf_doc, length = self.doc_tf[i], self.doc_len[i]
        breakdown: dict[str, float] = {}
        for term in query_terms:
            tf = tf_doc.get(term, 0)
            if tf == 0:
                continue
            norm = self.k1 * (1 - self.b + self.b * length / self.avgdl)
            breakdown[term] = breakdown.get(term, 0.0) + self.idf(term) * tf * (self.k1 + 1) / (tf + norm)
        return ScoredDoc(index=i, score=sum(breakdown.values()), breakdown=breakdown)

    def search(self, query_terms: list[str], top_k: int = 5) -> list[ScoredDoc]:
        scored = [self.score_doc(query_terms, i) for i in range(self.n_docs)]
        scored = [s for s in scored if s.score > 0]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]


class TfIdf(BM25):
    """Classic TF-IDF: no saturation, no length normalisation. Great for seeing why BM25 exists."""

    def score_doc(self, query_terms: list[str], i: int) -> ScoredDoc:
        tf_doc = self.doc_tf[i]
        breakdown: dict[str, float] = {}
        for term in query_terms:
            tf = tf_doc.get(term, 0)
            if tf:
                breakdown[term] = breakdown.get(term, 0.0) + tf * self.idf(term)
        return ScoredDoc(index=i, score=sum(breakdown.values()), breakdown=breakdown)
