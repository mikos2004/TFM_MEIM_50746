import os
import requests
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

EMB_MODEL = os.getenv("EMB_MODEL", "snowflake-arctic-embed2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def get_embeddings(texts: list[str], model: str = EMB_MODEL) -> np.ndarray:
    """Obtém embeddings em batch via API do Ollama (/api/embed)."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return np.array(data["embeddings"], dtype=np.float32)


# Frases
reference = "The capital of Portugal is Lisbon."
prediction = "Lisbon is the capital of Portugal."

# Embeddings em batch (uma só chamada HTTP)
embeddings = get_embeddings([reference, prediction])

# Forma: (2, dim) → ex.: (2, 1024) para snowflake-arctic-embed2
print(f"Shape: {embeddings.shape}")

# Similaridade de cosseno entre as duas frases
similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])
print(f"Cosine Similarity: {similarity[0][0]:.4f}")