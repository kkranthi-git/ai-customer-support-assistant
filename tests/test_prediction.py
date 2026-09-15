from src.pipeline import build_context
from src.retriever import retrieve


def test_build_context():
    results = [
        {
            "source": "payments.md",
            "text": "Payment failed. Check your payment details."
        },
        {
            "source": "billing.md",
            "text": "Billing information can be updated."
        }
    ]

    context = build_context(results)

    assert "payments.md" in context
    assert "billing.md" in context
    assert "Payment failed" in context


def test_retrieve_returns_results():
    class MockModel:
        def encode(self, texts):
            import numpy as np
            return np.array([[1.0, 0.0]])

    import numpy as np
    import pandas as pd

    chunks_df = pd.DataFrame({
        "source": ["payments.md", "billing.md"],
        "chunk_id": [0, 0],
        "text": [
            "Payment failed. Check your payment method.",
            "Billing information and invoices."
        ]
    })

    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    results = retrieve(
        query="payment failed",
        model=MockModel(),
        chunks_df=chunks_df,
        embeddings=embeddings,
        top_k=1,
        min_similarity=0.40
    )

    assert len(results) == 1
    assert results[0]["source"] == "payments.md"


def test_retrieve_returns_empty_for_irrelevant_query():
    class MockModel:
        def encode(self, texts):
            import numpy as np
            return np.array([[0.0, 1.0]])

    import numpy as np
    import pandas as pd

    chunks_df = pd.DataFrame({
        "source": ["payments.md"],
        "chunk_id": [0],
        "text": ["Payment failed."]
    })

    embeddings = np.array([
        [1.0, 0.0]
    ])

    results = retrieve(
        query="delivery address",
        model=MockModel(),
        chunks_df=chunks_df,
        embeddings=embeddings,
        top_k=3,
        min_similarity=0.40
    )

    assert results == []