"""Local in-memory vector search backend."""

from __future__ import annotations

import numpy as np


def search_local_matrix(
    node_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Search by multiplying normalized query vectors by all node vectors.

    With L2-normalized vectors, this product is cosine similarity.
    """
    outputs = []
    for query_embedding in query_embeddings:
        scores = node_embeddings @ query_embedding
        order = np.argsort(-scores)[:top_k]
        outputs.append((order, scores[order]))
    return outputs
