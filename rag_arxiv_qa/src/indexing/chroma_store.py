from typing import List, Dict, Any
import chromadb


class ChromaVectorStore:
    def __init__(self, config: dict):
        db_cfg = config["vector_db"]

        self.persist_dir = db_cfg["persist_directory"]
        self.collection_name = db_cfg["collection_name"]
        self.expected_dimension = config["embeddings"]["dimension"]

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
        )

        # Check if collection exists and has correct dimension
        try:
            existing_collection = self.client.get_collection(name=self.collection_name)
            # Check dimension by trying to query with a test embedding
            try:
                test_emb = [0.0] * self.expected_dimension
                existing_collection.query(
                    query_embeddings=[test_emb],
                    n_results=1
                )
                # If query succeeds, dimension matches
                self.collection = existing_collection
            except Exception:
                # Dimension mismatch - delete and recreate
                print(f"Deleting existing collection with wrong dimension...")
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
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