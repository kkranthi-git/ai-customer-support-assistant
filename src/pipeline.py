from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import load_embedding_model
from .generator import create_client, generate_answer
from .retriever import retrieve


def build_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"Source: {result['source']}\n{result['text']}"
        for result in results
    )


def answer_question(
    query: str,
    chunks_df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_model,
    client,
    top_k: int = 3
) -> dict:

    results = retrieve(
        query=query,
        model=embedding_model,
        chunks_df=chunks_df,
        embeddings=embeddings,
        top_k=top_k
    )

    if not results:
        return {
            "answer": (
                "I don't have enough information in the knowledge base "
                "to answer this question."
            ),
            "sources": [],
            "results": []
        }

    context = build_context(results)

    answer = generate_answer(
        query=query,
        context=context,
        client=client
    )

    sources = list(
        dict.fromkeys(
            result["source"] for result in results
        )
    )

    return {
        "answer": answer,
        "sources": sources,
        "results": results
    }


def load_pipeline():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    chunks_df = pd.read_csv(
        processed_dir / "chunks.csv"
    )

    embeddings = np.load(
        processed_dir / "embeddings.npy"
    )

    embedding_model = load_embedding_model()
    client = create_client()

    return (
        chunks_df,
        embeddings,
        embedding_model,
        client
    )