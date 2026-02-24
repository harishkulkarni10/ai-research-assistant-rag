# FastAPI application for RAG service
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from rag_arxiv_qa.src.api.routes import router
from rag_arxiv_qa.src.utils.logger import setup_logging
from rag_arxiv_qa.src.utils.metrics import get_metrics

setup_logging(level="INFO")

app = FastAPI(
    title="ArXiv Research Assistant API",
    description="RAG system for querying ArXiv research papers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "ArXiv Research Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    metrics_collector = get_metrics()
    return metrics_collector.get_summary()


if __name__ == "__main__":
    uvicorn.run(
        "rag_arxiv_qa.src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
