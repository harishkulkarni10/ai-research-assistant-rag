from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    """
    Wrapper around ChromaDB.

    Responsibilities:
    - Initialize persistent Chroma client
    - Create or load a collection
    - Upsert embeddings with metadata
    - Query embeddings 

    """

    def __init__(self, config: Dict[str, Any]):
        db_cfg = config["vector_db"]

        self.persist_directory = db_cfg["persist_directory"]
        self.collection_name = db_cfg["collection_name"]

        self.client = chromadb.Client(
            Settings(
                persist_directory=self.persist_directory
            )
        )

        # Create or load collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        *,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Idempotent upsert of vectors into Chroma.
        """
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        *,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store.

        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

    def count(self) -> int:
        """Return number of stored vectors."""
        return self.collection.count()