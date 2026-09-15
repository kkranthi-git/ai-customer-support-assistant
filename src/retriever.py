import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def retrieve(
    query: str,
    model,
    chunks_df,
    embeddings: np.ndarray,
    top_k: int = 3,
    min_similarity: float = 0.40
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:

        similarity = float(similarities[index])

        if similarity < min_similarity:
            continue

        results.append({
            "source": chunks_df.iloc[index]["source"],
            "chunk_id": int(chunks_df.iloc[index]["chunk_id"]),
            "similarity": similarity,
            "text": chunks_df.iloc[index]["text"]
        })

    return results