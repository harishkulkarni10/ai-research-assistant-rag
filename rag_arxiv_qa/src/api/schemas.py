"""
API request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    """Query request schema."""
    query: str = Field(..., description="User query", min_length=1, max_length=1000)


class Citation(BaseModel):
    """Citation schema."""
    source: str
    position: Optional[str] = None
    text: Optional[str] = None


class QueryResponse(BaseModel):
    """Query response schema."""
    answer: str
    citations: List[Any] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
