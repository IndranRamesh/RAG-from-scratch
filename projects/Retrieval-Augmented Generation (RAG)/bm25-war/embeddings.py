from sentence_transformers import SentenceTransformer
from typing import List, Union

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Convert a list of texts into embedding vectors."""
    try:
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    except Exception as e:
        print(f"Error during embedding: {e}")
        return []  # Always return a list , even of failure
