"""Clustering utilities for grouping exemplar attack classes.

Uses scikit-learn KMeans when available. This is optional and the
module will raise a friendly error if sklearn is not installed.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

try:
    from sklearn.cluster import KMeans
    import numpy as np
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


class Clusterer:
    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_to_label = {}
        self.centers = None

    def fit(self, exemplars: Dict[str, List[str]], semantic_engine) -> None:
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for clustering")

        texts = []
        labels = []
        for label, exs in exemplars.items():
            for ex in exs:
                texts.append(ex)
                labels.append(label)

        X = semantic_engine.embed(texts)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X)
        self.centers = self.kmeans.cluster_centers_
        preds = list(self.kmeans.labels_)

        # map cluster -> majority exemplar label
        from collections import Counter

        for cid in set(preds):
            idxs = [i for i, p in enumerate(preds) if p == cid]
            cnt = Counter(labels[i] for i in idxs)
            self.cluster_to_label[cid] = cnt.most_common(1)[0][0]

    def predict(self, text: str, semantic_engine) -> Tuple[str | None, float]:
        """Predict cluster label and similarity score (0..1).

        Returns (label, score). If clusterer is not fitted returns (None, 0.0).
        """
        if self.kmeans is None or self.centers is None:
            return None, 0.0

        x = semantic_engine.embed([text])[0]
        cid = int(self.kmeans.predict([x])[0])
        label = self.cluster_to_label.get(cid, None)

        # cosine similarity between x and center
        import numpy as np

        center = self.centers[cid]
        denom = (np.linalg.norm(x) * np.linalg.norm(center))
        sim = float(np.dot(x, center) / denom) if denom > 0 else 0.0
        sim_norm = max(0.0, min(1.0, (sim + 1.0) / 2.0))
        return label, sim_norm
