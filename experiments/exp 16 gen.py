"""
Semantic Similarity Search
--------------------------
Generates text embeddings and performs semantic similarity search
over a collection of documents using sentence-transformers.

Install dependencies first:
    pip install sentence-transformers numpy

Optional (faster search over large corpora):
    pip install faiss-cpu
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Loads a pretrained sentence embedding model.
        'all-MiniLM-L6-v2' is small, fast, and works well for general text.
        Other good options: 'all-mpnet-base-v2' (higher quality, slower),
        'multi-qa-MiniLM-L6-cos-v1' (tuned for question-answer retrieval).
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def index(self, documents: list[str]):
        """Embed and store a list of documents for later search."""
        self.documents = documents
        self.embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit-norm so dot product == cosine similarity
            show_progress_bar=False,
        )

    def search(self, query: str, top_k: int = 5):
        """Return the top_k most semantically similar documents to the query."""
        if self.embeddings is None:
            raise ValueError("No documents indexed yet. Call .index() first.")

        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # Cosine similarity via dot product (vectors are already normalized)
        scores = self.embeddings @ query_vec

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"document": self.documents[i], "score": float(scores[i])}
            for i in top_indices
        ]


if __name__ == "__main__":
    corpus = [
        "The cat sat on the warm windowsill in the afternoon sun.",
        "Stock markets rallied today after inflation data came in lower than expected.",
        "Machine learning models can learn patterns from large datasets.",
        "She adopted a rescue dog from the local animal shelter.",
        "The Federal Reserve is expected to discuss interest rate changes.",
        "Neural networks are inspired by the structure of the human brain.",
        "He went hiking in the mountains over the weekend.",
        "Quarterly earnings reports showed strong growth in tech companies.",
    ]

    searcher = SemanticSearcher()
    searcher.index(corpus)

    query = "How is the economy doing?"
    results = searcher.search(query, top_k=3)

    print(f"Query: {query}\n")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['document']}")
