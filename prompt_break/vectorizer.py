"""Pure-Python TF-IDF vectorizer (copied from advanced_jailbreak_agent)."""

import math
import re
from collections import Counter


class PureTFIDFVectorizer:
    """A lightweight TF-IDF vectorizer with cosine similarity."""

    TOKEN_RE = re.compile(r"[a-z0-9_]+")

    def __init__(self):
        self.vocabulary_ = {}
        self.idf_ = {}
        self.fitted_ = False

    def _tokenize(self, text):
        return self.TOKEN_RE.findall(text.lower())

    def fit(self, documents):
        doc_count = len(documents)
        if doc_count == 0:
            self.vocabulary_ = {}
            self.idf_ = {}
            self.fitted_ = True
            return self

        df_counter = Counter()
        for doc in documents:
            terms = set(self._tokenize(doc))
            for term in terms:
                df_counter[term] += 1

        self.vocabulary_ = {term: i for i, term in enumerate(sorted(df_counter))}
        self.idf_ = {}
        for term, df in df_counter.items():
            self.idf_[term] = math.log((1 + doc_count) / (1 + df)) + 1.0

        self.fitted_ = True
        return self

    def transform(self, documents):
        if not self.fitted_:
            raise ValueError("Vectorizer must be fitted before calling transform().")

        vocab_size = len(self.vocabulary_)
        vectors = []
        for doc in documents:
            tokens = self._tokenize(doc)
            term_counts = Counter(tokens)
            total_terms = float(len(tokens)) if tokens else 1.0
            vec = [0.0] * vocab_size
            for term, count in term_counts.items():
                idx = self.vocabulary_.get(term)
                if idx is None:
                    continue
                tf = count / total_terms
                vec[idx] = tf * self.idf_.get(term, 1.0)
            vectors.append(vec)
        return vectors

    def cosine_similarity(self, vec_a, vec_b):
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec_a, vec_b):
            dot += a * b
            norm_a += a * a
            norm_b += b * b
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
