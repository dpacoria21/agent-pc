"""FAISS vector search backend."""

from __future__ import annotations

import numpy as np


def search_faiss_index_flat_ip(
    node_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Search with FAISS IndexFlatIP.

    IndexFlatIP is exact inner-product search. Because the comparison pipeline
    passes L2-normalized embeddings, inner product is equivalent to cosine
    similarity.
    """
    import faiss

    index = faiss.IndexFlatIP(node_embeddings.shape[1])
    index.add(node_embeddings)
    scores, indices = index.search(query_embeddings, top_k)
    return [(indices[i], scores[i]) for i in range(len(query_embeddings))]
