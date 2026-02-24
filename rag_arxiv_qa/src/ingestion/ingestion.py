from typing import Iterable, Dict, Any
from tqdm import tqdm 
import gc

from rag_arxiv_qa.src.chunking.chunker import Chunker
from rag_arxiv_qa.src.embeddings.embedder import Embedder
from rag_arxiv_qa.src.indexing.chroma_store import ChromaVectorStore

class IngestionPipeline:
    """
    Streaming ingestion pipeline
    doc -> chunk -> embed -> store
    """
    def __init__(self, config: Dict[str, Any]):
        self.chunker = Chunker(config)
        self.embedder = Embedder(config)
        self.vector_store = ChromaVectorStore(config)

        # Process one document at a time to avoid segfaults
        self.embed_batch_size = 4

    def ingest(self, documents: Iterable[Dict[str, Any]]) -> None:
        """
        Run ingestion over a stream of documents.
        Processes one chunk at a time to avoid segfaults on Windows.
        """
        processed_count = 0
        error_count = 0
        total_chunks = 0

        for doc in tqdm(documents, desc="Ingesting documents"):
            try:
                # Process one document at a time: chunk -> embed -> store
                chunks = list(self.chunker.chunk_document(
                    doc_id=doc["doc_id"],
                    text=doc["text"],
                    base_metadata=doc["metadata"],
                ))
                
                if not chunks:
                    continue
                
                # Process ONE chunk at a time to avoid segfaults
                for chunk in chunks:
                    try:
                        # Embed single chunk
                        embedding = self.embedder.embed_documents([chunk["text"]])
                        
                        # Store immediately
                        self.vector_store.upsert(
                            ids=[chunk["chunk_id"]],
                            embeddings=embedding.tolist(),
                            documents=[chunk["text"]],
                            metadatas=[chunk["metadata"]],
                        )
                        
                        total_chunks += 1
                        
                        # Aggressive cleanup after each chunk
                        del embedding
                        gc.collect()
                        
                    except Exception as e:
                        print(f"\nWarning: Failed to process chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                        gc.collect()
                        continue
                
                processed_count += 1
                
                # Force garbage collection after each document
                gc.collect()
                
            except Exception as e:
                error_count += 1
                print(f"\nWarning: Failed to process document {doc.get('doc_id', 'unknown')}: {e}")
                gc.collect()
                continue
        
        print(f"\nCompleted: {processed_count} documents, {total_chunks} chunks")
        if error_count > 0:
            print(f"Errors: {error_count} documents failed")
