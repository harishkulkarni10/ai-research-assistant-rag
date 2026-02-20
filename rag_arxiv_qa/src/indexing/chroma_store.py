from typing import List, Dict, Any
import chromadb


class ChromaVectorStore:
    """
    ChromaDB vector store.
    """

    def __init__(self, config: dict):
        db_cfg = config["vector_db"]

        self.persist_dir = db_cfg["persist_directory"]
        self.collection_name = db_cfg["collection_name"]

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embedding: List[float], top_k: int):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def count(self) -> int:
        return self.collection.count()