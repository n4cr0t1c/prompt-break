"""Semantic similarity utilities (sentence-transformers optional).

If `sentence-transformers` is available the engine will use it for dense
embeddings; otherwise it falls back to the repository's TF-IDF vectorizer.
All heavy dependencies are optional — the code gracefully degrades.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

try:
    from .vectorizer import PureTFIDFVectorizer
    import numpy as np
    _TFIDF_AVAILABLE = True
except Exception:
    _TFIDF_AVAILABLE = False


class SemanticEngine:
    """Encapsulates embedding and exemplar matching.

    - If `sentence-transformers` is present uses a lightweight transformer
      (default `all-MiniLM-L6-v2`).
    - Otherwise uses the local `PureTFIDFVectorizer` as a fallback.
    """

    def __init__(self, model_name: str | None = None):
        self.available = False
        self.use_st = False
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self._examples_emb = None
        self._examples = []

        if _ST_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.use_st = True
                self.available = True
            except Exception:
                logging.exception("Failed to load sentence-transformers model; falling back.")
                self.model = None

        if not self.use_st and _TFIDF_AVAILABLE:
            self.vectorizer = PureTFIDFVectorizer()
            self.available = True

    def fit_examples(self, exemplars: Dict[str, List[str]]):
        """Fit / encode exemplar sentences from the mapping {label: [examples]}."""
        flat = []
        self._labels = []
        for label, exs in exemplars.items():
            for ex in exs:
                flat.append(ex)
                self._labels.append(label)
        self._examples = flat
        if self.use_st:
            import numpy as np

            self._examples_emb = self.model.encode(flat, convert_to_numpy=True, normalize_embeddings=True)
        else:
            # TF-IDF fallback — fit the vocabulary and cache vectors
            self.vectorizer.fit(flat)
            vecs = self.vectorizer.transform(flat)
            import numpy as np

            # convert list-of-lists to 2D numpy array
            self._examples_emb = np.array([np.array(v) for v in vecs])

    def embed(self, texts: List[str]):
        """Return embeddings for the given texts as a 2D numpy array."""
        if not self.available:
            raise RuntimeError("No semantic backend available")
        import numpy as np

        if self.use_st:
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        vecs = self.vectorizer.transform(texts)
        return np.array([np.array(v) for v in vecs])

    def most_similar_to_exemplars(self, text: str) -> Tuple[str | None, float, str | None]:
        """Return (best_label, score(0..1), best_example_text).

        Score is cosine similarity in the 0..1 range (higher is more similar).
        """
        if not self.available:
            return None, 0.0, None
        if not self._examples_emb or not self._examples:
            # try to load exemplars from the lightweight agent
            try:
                from jailbreak_agent import EXEMPLARS  # type: ignore

                self.fit_examples(EXEMPLARS)
            except Exception:
                return None, 0.0, None

        emb = self.embed([text])[0]
        import numpy as np

        # cosine similarity
        dots = np.dot(self._examples_emb, emb)
        norms = (np.linalg.norm(self._examples_emb, axis=1) * np.linalg.norm(emb))
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = np.where(norms > 0, dots / norms, 0.0)

        # normalize to 0..1
        sims = (sims + 1.0) / 2.0
        best_idx = int(np.argmax(sims))
        best_score = float(max(0.0, min(1.0, sims[best_idx])))
        best_label = self._labels[best_idx] if hasattr(self, "_labels") else None
        best_example = self._examples[best_idx]
        return best_label, best_score, best_example
