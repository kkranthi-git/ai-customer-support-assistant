# AI-Powered Customer Support Assistant

An AI-powered customer support assistant that uses semantic search, embeddings, and a Large Language Model (LLM) to retrieve relevant knowledge and generate grounded customer support responses.

The system follows a Retrieval-Augmented Generation (RAG) architecture to reduce unsupported answers by providing the LLM with relevant information from a controlled knowledge base.

---

## Project Overview

Traditional customer support systems often rely on keyword-based search or predefined responses.

This project builds a more intelligent support assistant that can understand the semantic meaning of a customer's question, retrieve relevant information from a knowledge base, and generate a concise response using an LLM.

### Example

**Customer Question**

> My payment failed. What should I do?

**Retrieved Knowledge**

> Payment information and troubleshooting steps from `payments.md`

**Generated Response**

> Please check your payment details and try the payment again. If the issue continues, contact customer support.

**Source**

> `payments.md`

---

## Architecture

```text
Customer Question
       |
       v
Text Embedding
       |
       v
Semantic Search
       |
       v
Relevant Knowledge
       |
       v
Context Construction
       |
       v
Large Language Model
       |
       v
Grounded Answer
       |
       v
Sources

## Key Features
Semantic search using sentence embeddings
Vector similarity retrieval
Retrieval-Augmented Generation (RAG)
Hugging Face hosted LLM
Grounded responses using retrieved knowledge
Source attribution
Relevance threshold for irrelevant questions
Streamlit web application
Unit tests with pytest
Docker support
Environment-variable based API authentication

## Technology Stack

### Programming
- Python

### Machine Learning / NLP
-Sentence Transformers
-all-MiniLM-L6-v2
-Scikit-learn
-Cosine similarity

### Vector Search
-FAISS

### Generative AI
-Hugging Face Inference API
-Meta Llama 3.1 8B Instruct

### Application
- Streamlit

### Testing
- Pytest

### Deployment
- Docker

Knowledge Base
The assistant currently uses a small customer-support knowledge base containing:
data/
└── knowledge_base/
    ├── account.md
    ├── billing.md
    ├── payments.md
    ├── refunds.md
    ├── returns.md
    └── technical_support.md
These documents are divided into smaller chunks before generating embeddings.

RAG Pipeline

1. Document Loading

Markdown documents are loaded from the knowledge base.

2. Chunking

Documents are divided into smaller overlapping chunks.

Current configuration:

Chunk size: 500 characters
Chunk overlap: 100 characters

3. Embedding Generation

Each chunk is converted into a numerical vector using:

all-MiniLM-L6-v2

The resulting embeddings contain 384 dimensions.

4. Semantic Retrieval

When a customer asks a question:

The question is converted into an embedding.
Cosine similarity is calculated against the knowledge-base embeddings.
The most relevant chunks are retrieved.
A similarity threshold filters out weak matches.

Current retrieval threshold:
0.40

5. Context Construction
Retrieved chunks are combined with their source filenames and provided to the LLM as context.

6. LLM Generation
The LLM generates an answer using only the retrieved context.

The system instructs the model not to invent information when the knowledge base does not contain enough information.

7. Source Attribution
The application displays the source documents used to generate the response.

Evaluation

Retrieval performance was evaluated using five manually labeled customer-support questions.

Metric	Result
Retrieval Accuracy	100%
Test Questions	5

Test questions included:

Password reset
Product returns
Refund duration
Internet connection problems
Duplicate billing/payment

The evaluation checks whether the expected knowledge-base document is retrieved for each question.

Handling Unknown Questions

The system includes a similarity threshold to avoid passing unrelated information to the LLM.

For example:

Question:
Can I change my delivery address?

If the knowledge base does not contain sufficiently relevant information, the system responds:

I don't have enough information in the knowledge base
to answer this question.

This provides a basic grounding and hallucination-control mechanism.

Project Structure
ai-customer-support-assistant/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── knowledge_base/
│   │   ├── account.md
│   │   ├── billing.md
│   │   ├── payments.md
│   │   ├── refunds.md
│   │   ├── returns.md
│   │   └── technical_support.md
│   │
│   ├── processed/
│   │   ├── chunks.csv
│   │   └── embeddings.npy
│   │
│   └── raw/
│
├── models/
│
├── notebooks/
│   ├── 01_knowledge_base.ipynb
│   ├── 02_chunking.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_retrieval.ipynb
│   ├── 05_rag_generation.ipynb
│   └── 06_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── generator.py
│   ├── pipeline.py
│   └── predict.py
│
├── tests/
│   └── test_prediction.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md

Installation

Clone the repository:

git clone https://github.com/kkranthi-git/ai-customer-support-assistant.git
cd ai-customer-support-assistant

Create a virtual environment:

python -m venv .venv

Activate the environment on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Environment Variables

The project uses a Hugging Face API token for LLM generation.

Create a .env file in the project root:

HF_TOKEN=your_huggingface_token

Never commit .env or your API token to GitHub.

Run the Application

Start the Streamlit application:

streamlit run app/streamlit_app.py

The application will open in your browser.

Run Tests

Run the test suite:

pytest -q

Current test coverage includes:

Context construction
Relevant document retrieval
Irrelevant query rejection

Expected result:

3 passed
Docker

Build the Docker image:

docker build -t ai-customer-support-assistant .

Run the container:

docker run -p 8501:8501 --env-file .env ai-customer-support-assistant

Open:

http://localhost:8501

The Docker image contains the application code and processed knowledge-base data but does not contain the .env file or raw data.

Example Questions

Try questions such as:

I forgot my password. How can I reset it?
My payment failed. What should I do?
How long does a refund take?
I want to return a product. What is the process?
My internet is not working.

Engineering Design Decisions
Why embeddings?

Keyword matching can fail when two questions use different words but have similar meanings.

Embeddings allow the system to compare the semantic meaning of the customer's question with the knowledge-base content.

Why RAG?

Instead of expecting the LLM to know all customer-support information, the system retrieves relevant information from a controlled knowledge base and provides it to the model.

This makes responses more grounded in the available information.

Why a similarity threshold?

Retrieving weakly related documents can provide misleading context to the LLM.

A similarity threshold helps the system reject questions when the retrieved information is not sufficiently relevant.

Why source attribution?

Displaying source documents improves transparency and allows users to understand where the response originated.

Limitations
The current knowledge base is small and synthetic.
Retrieval evaluation uses a limited manually labeled test set.
The system uses a fixed similarity threshold.
The current embedding model is relatively lightweight.
The LLM depends on Hugging Face Inference availability and token limits.
The system does not yet maintain multi-turn conversation memory.
Future Improvements

Potential improvements include:

Larger production knowledge base
Better chunking strategies
Hybrid keyword + vector retrieval
Reranking retrieved documents
Evaluation with larger benchmark datasets
Retrieval metrics such as Precision@K and Recall@K
LLM response-quality evaluation
Conversation memory
Streaming responses
Authentication
Monitoring and logging
Production vector database
Transformer-based reranking
Advanced RAG techniques
What I Learned

This project demonstrates an end-to-end Generative AI workflow:

Data
 ↓
Document Processing
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Retrieval
 ↓
Context Construction
 ↓
LLM
 ↓
Grounded Response
 ↓
Evaluation
 ↓
Deployment

The project helped develop practical experience with:

NLP
Embeddings
Semantic Search
Vector Retrieval
RAG
LLM integration
Prompt design
Hallucination control
Evaluation
Streamlit
Docker
Testing
Production-oriented Python project structure

Author
Kranthi Kumar
Data Scientist | AI Engineer
