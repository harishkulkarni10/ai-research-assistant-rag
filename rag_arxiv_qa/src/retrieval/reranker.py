from typing import List, Dict, Any

import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Cross-encoder re-ranker.

    Takes a query and retrieved chunks, scores (query, chunk) pairs,
    and returns the most relevant chunks.
    """

    def __init__(self, config: Dict[str, Any]):
        rerank_cfg = config["reranking"]

        self.model_name = rerank_cfg["model"]
        self.top_k = rerank_cfg.get("top_k", 10)
        self.batch_size = rerank_cfg.get("batch_size", 16)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CrossEncoder(
            self.model_name,
            device=device,
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Re-rank retrieved chunks using a cross-encoder.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
        )

        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        candidates.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return candidates[: self.top_k]