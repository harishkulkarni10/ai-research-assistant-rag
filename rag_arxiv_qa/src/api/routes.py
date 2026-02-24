# API routes for RAG service
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional

from rag_arxiv_qa.src.retrieval.pipeline import RetrievalPipeline
from rag_arxiv_qa.src.generation.generator import Generator
from rag_arxiv_qa.src.services.rag_service import RAGService
from rag_arxiv_qa.src.utils.config import load_config

router = APIRouter(prefix="/api/v1", tags=["rag"])

_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        config = load_config()
        retrieval_pipeline = RetrievalPipeline(config)
        generator = Generator(config)
        _rag_service = RAGService(retrieval_pipeline, generator)
    return _rag_service


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query about research papers", min_length=1, max_length=1000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in the world of language models?"
            }
        }


class QueryResponse(BaseModel):
    answer: str
    citations: list
    confidence_score: float
    metadata: dict


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
):
    try:
        response = service.answer(request.query)
        return QueryResponse(
            answer=response.answer,
            citations=response.citations,
            confidence_score=response.confidence_score,
            metadata=response.metadata,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
