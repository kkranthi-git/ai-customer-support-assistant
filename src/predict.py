from .pipeline import answer_question, load_pipeline


def predict(query: str) -> dict:
    (
        chunks_df,
        embeddings,
        embedding_model,
        client
    ) = load_pipeline()

    return answer_question(
        query=query,
        chunks_df=chunks_df,
        embeddings=embeddings,
        embedding_model=embedding_model,
        client=client,
        top_k=3
    )