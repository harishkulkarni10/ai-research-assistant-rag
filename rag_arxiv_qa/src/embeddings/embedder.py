from typing import Dict
import numpy as np

from sentence_transformers import SentenceTransformer

class Embedder:
    """
    - Load embedding model
    - Handle batching 
    """
    def __init__(self, config: dict):
        emb_cfg = config['embeddings']

        self.model_name = emb_cfg['model']
        self.device = emb_cfg.get("device", "cpu")
        self.batch_size = emb_cfg.get("batch_size", 32)
        self.normalize = emb_cfg.get("normalize", True)

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of document texts
        """
        embeddings = self.model.encode(
            texts, 
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query 
        """
        embedding = self.model.encode(
            text,
            batch_size = 1,
            show_progress_bar = False,
            convert_to_numpy = True,
            normalize_embeddings = self.normalize,
        )

        return embedding
