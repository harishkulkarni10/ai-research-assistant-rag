import time
import uuid

from rag_arxiv_qa.src.services.schemas import RAGResponse
from rag_arxiv_qa.src.utils.logger import log_event, log_error
from rag_arxiv_qa.src.utils.metrics import get_metrics


class RAGService:

    def __init__(self, retrieval_pipeline, generator):
        self.retrieval_pipeline = retrieval_pipeline
        self.generator = generator
        self.metrics = get_metrics()

    def answer(self, query: str) -> RAGResponse:
        request_id = str(uuid.uuid4())
        total_start = time.perf_counter()
        
        # Track request
        self.metrics.increment("rag_requests_total")
        
        try:
            # -----------------------------
            # Retrieval
            # -----------------------------
            retrieval_start = time.perf_counter()
            retrieved_chunks = self.retrieval_pipeline.retrieve(query)
            retrieval_time = time.perf_counter() - retrieval_start
            
            # Record retrieval metrics
            self.metrics.record_latency("retrieval", retrieval_time)
            self.metrics.set_gauge("retrieved_chunks", len(retrieved_chunks))

            # -----------------------------
            # Generation
            # -----------------------------
            generation_start = time.perf_counter()
            generation_output = self.generator.generate(
                query=query,
                chunks=retrieved_chunks,
            )
            generation_time = time.perf_counter() - generation_start
            
            # Record generation metrics
            self.metrics.record_latency("generation", generation_time)

            total_time = time.perf_counter() - total_start
            self.metrics.record_latency("rag_total", total_time)

            metadata = {
                "request_id": request_id,
                "retrieved_chunks": len(retrieved_chunks),
                "retrieval_time_sec": round(retrieval_time, 4),
                "generation_time_sec": round(generation_time, 4),
                "total_time_sec": round(total_time, 4),
            }

            log_event("rag_request", {
                "request_id": request_id,
                "retrieved_chunks": len(retrieved_chunks),
                "retrieval_time_sec": retrieval_time,
                "generation_time_sec": generation_time,
                "total_time_sec": total_time,
            })
            
            # Track success
            self.metrics.increment("rag_requests_success")

            return RAGResponse(
                answer=generation_output.get("answer", ""),
                citations=generation_output.get("citations", []),
                confidence_score=generation_output.get("confidence_score", 0.0),
                metadata=metadata,
            )
        
        except Exception as e:
            # Track errors
            self.metrics.increment("rag_requests_errors")
            log_error(e, {"request_id": request_id, "query": query[:100]})
            raise