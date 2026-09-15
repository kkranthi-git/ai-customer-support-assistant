import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.pipeline import load_pipeline, answer_question


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Customer Support Assistant",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# Load RAG pipeline
# --------------------------------------------------

@st.cache_resource
def initialize_pipeline():
    return load_pipeline()


try:
    (
        chunks_df,
        embeddings,
        embedding_model,
        client
    ) = initialize_pipeline()

except Exception as e:
    st.error("Failed to initialize the AI pipeline.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# User interface
# --------------------------------------------------

st.title("🤖 AI Customer Support Assistant")

st.write(
    "Ask a customer support question and get an AI-generated "
    "answer grounded in the knowledge base."
)

st.divider()


query = st.text_input(
    "Customer Question",
    placeholder="e.g. My payment failed. What should I do?"
)


# --------------------------------------------------
# Generate answer
# --------------------------------------------------

if st.button("Get Answer", type="primary"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching knowledge base and generating answer..."):

            try:

                result = answer_question(
                    query=query,
                    chunks_df=chunks_df,
                    embeddings=embeddings,
                    embedding_model=embedding_model,
                    client=client,
                    top_k=3
                )

            except Exception as e:

                st.error("Something went wrong while generating the answer.")
                st.exception(e)
                st.stop()


        # --------------------------------------------------
        # Answer
        # --------------------------------------------------

        st.subheader("Answer")

        st.write(result["answer"])


        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        if result["sources"]:
            st.subheader("Sources")

            for source in result["sources"]:
                st.write(f"📄 {source}")


        # --------------------------------------------------
        # Retrieved context
        # --------------------------------------------------

        with st.expander("View retrieved knowledge"):

            for result_item in result["results"]:

                st.markdown(
                    f"**Source:** {result_item['source']}"
                )

                st.markdown(
                    f"**Similarity:** "
                    f"{result_item['similarity']:.4f}"
                )

                st.write(result_item["text"])

                st.divider()