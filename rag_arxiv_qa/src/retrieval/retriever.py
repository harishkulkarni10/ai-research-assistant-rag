from typing import Dict, List, Any

from rag_arxiv_qa.src.embeddings.embedder import Embedder
from rag_arxiv_qa.src.indexing.chroma_store import ChromaVectorStore


class DenseRetriever:
    """
    Dense vector retriever using ChromaDB.
    - embed query
    - perform similarity search
    - return top-k chunks
    """

    def __init__(self, config: Dict[str, Any]):
        self.embedder = Embedder(config)
        self.vector_store = ChromaVectorStore(config)
        self.top_k = config["retrieval"].get("top_k", 50)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks for a query.
        """

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        candidates = []

        if not results["documents"]:
            return []

        for idx in range(len(results["documents"][0])):
            candidates.append({
                "chunk_id": results["ids"][0][idx],
                "text": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "score": results["distances"][0][idx],
            })

        return candidates