from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer
):
    if not texts:
        raise ValueError("No texts provided for embedding.")

    return model.encode(
        texts,
        show_progress_bar=True
    )