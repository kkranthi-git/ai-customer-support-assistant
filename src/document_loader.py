from pathlib import Path


def load_documents(knowledge_base_dir: Path) -> list[dict]:
    documents = []

    for file_path in sorted(knowledge_base_dir.glob("*.md")):
        content = file_path.read_text(encoding="utf-8").strip()

        if content:
            documents.append({
                "source": file_path.name,
                "text": content
            })

    if not documents:
        raise ValueError("No knowledge-base documents found.")

    return documents