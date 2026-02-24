from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class RAGResponse:
    answer: str
    citations: List[int]
    confidence_score: float
    metadata: Dict[str, Any]