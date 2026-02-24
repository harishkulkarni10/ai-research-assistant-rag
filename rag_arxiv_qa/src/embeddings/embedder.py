from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, config: dict):
        emb_cfg = config["embeddings"]

        self.model_name = emb_cfg["model"]
        self.device = emb_cfg.get("device", "cpu")
        self.batch_size = emb_cfg.get("batch_size", 32)
        self.normalize = emb_cfg.get("normalize", True)

        # Heavy object: load once
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of document texts.
        """
        # Use very small batch size to avoid segfaults on Windows
        effective_batch_size = min(self.batch_size, 4)
        
        # Process one at a time if batch is very small to avoid issues
        if len(texts) <= 2:
            embeddings_list = []
            for text in texts:
                emb = self.model.encode(
                    [text],
                    batch_size=1,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize,
                )
                embeddings_list.append(emb[0])
            return np.array(embeddings_list)
        
        embeddings = self.model.encode(
            texts,
            batch_size=effective_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        """
        embedding = self.model.encode(
            query,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return embedding.tolist()