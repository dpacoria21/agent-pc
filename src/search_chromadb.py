"""ChromaDB vector search backend."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def search_chromadb_ephemeral(
    nodes: pd.DataFrame,
    documents: list[str],
    node_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Search with an ephemeral ChromaDB collection.

    The embeddings are supplied explicitly, so ChromaDB is used as a vector
    store/search backend rather than as an embedding model provider.
    """
    import chromadb
    from chromadb.config import Settings

    try:
        client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    except Exception:
        client = chromadb.Client(Settings(anonymized_telemetry=False))

    collection = client.create_collection(
        name=f"cp_nodes_backend_comparison_{int(time.time() * 1000)}",
        metadata={"hnsw:space": "cosine"},
    )
    ids = nodes["node_id"].astype(str).tolist()
    metadatas = [
        {
            "global_problem_id": str(row["global_problem_id"]),
            "node_type": str(row["node_type"]),
            "source": str(row.get("source", "")),
        }
        for _, row in nodes.iterrows()
    ]
    collection.add(
        ids=ids,
        embeddings=node_embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    result = collection.query(
        query_embeddings=query_embeddings.tolist(),
        n_results=top_k,
        include=["distances"],
    )

    outputs = []
    id_to_index = {node_id: idx for idx, node_id in enumerate(ids)}
    for ids_row, distances_row in zip(result["ids"], result["distances"]):
        indices = np.array([id_to_index[node_id] for node_id in ids_row], dtype=int)
        scores = np.array([max(0.0, 1.0 - float(distance)) for distance in distances_row], dtype=float)
        outputs.append((indices, scores))
    return outputs
