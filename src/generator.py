import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def create_client():
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN is not configured.")

    return InferenceClient(token=hf_token)


def generate_answer(
    query: str,
    context: str,
    client: InferenceClient
) -> str:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful customer support assistant. "
                "Answer only using the provided context. "
                "Do not make up information. "
                "If the context does not contain enough information, "
                "say that you do not have enough information."
            )
        },
        {
            "role": "user",
            "content": f"""
Customer question:
{query}

Context:
{context}

Answer the customer clearly and concisely.
"""
        }
    ]

    response = client.chat_completion(
        messages=messages,
        model=MODEL_ID,
        max_tokens=200,
        temperature=0.2
    )

    return response.choices[0].message.content