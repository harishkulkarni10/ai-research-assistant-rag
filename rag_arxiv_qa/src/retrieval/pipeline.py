from typing import Dict, Any, List

from rag_arxiv_qa.src.retrieval.retriever import DenseRetriever
from rag_arxiv_qa.src.retrieval.reranker import CrossEncoderReranker


class RetrievalPipeline:
    """
    Encapsulates dense retrieval and re-ranking.
    """
    def __init__(self, config: Dict[str, Any]):
        self.retriever = DenseRetriever(config)
        self.reranker = CrossEncoderReranker(config)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        # Dense retrieval
        candidates = self.retriever.retrieve(query)

        if not candidates:
            return []

        # Re-ranking
        reranked = self.reranker.rerank(query, candidates)
        return reranked