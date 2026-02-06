from typing import Iterable, Dict, Any
from tqdm import tqdm 

from rag_arxiv_qa.src.chunking.chunker import Chunker
from rag_arxiv_qa.src.embeddings.embedder import Embedder
from rag_arxiv_qa.src.indexing.chroma_store import ChromaVectorStore

class IngestionPipeline:
    """
    Streaming ingestin pipeline
    doc -> chunk -> embed -> store
    """
    def __init__(self, config: Dict[str, Any]):
        self.chunker = Chunker(config)
        self.embedder = Embedder(config)
        self.vector_store = ChromaVectorStore(config)

        self.embed_batch_size = config["embeddings"].get("batch_size", 32)

    def ingest(self, documents: Iterable[Dict[str, Any]]) -> None:
        """
        Run ingestion over a stream of documents.

        Parameters
        ----------
        documents : Iterable[dict]
            Each document must have:
            - doc_id
            - text
            - metadata
        """

        buffer_chunks = []
        buffer_metadatas = []
        buffer_texts = []
        buffer_ids = []

        for doc in tqdm(documents, desc="Ingesting documents"):
            for chunk in self.chunker.chunk_document(
                doc_id=doc["doc_id"],
                text=doc["text"],
                base_metadata=doc["metadata"],
            ):
                buffer_texts.append(chunk["text"])
                buffer_ids.append(chunk["chunk_id"])
                buffer_metadatas.append(chunk["metadata"])
                buffer_chunks.append(chunk)

                if len(buffer_texts) >= self.embed_batch_size:
                    self._flush(
                        buffer_texts,
                        buffer_ids,
                        buffer_metadatas,
                    )
                    buffer_texts.clear()
                    buffer_ids.clear()
                    buffer_metadatas.clear()
                    buffer_chunks.clear()

        # Flush remaining chunks
        if buffer_texts:
            self._flush(
                buffer_texts,
                buffer_ids,
                buffer_metadatas,
            )

    def _flush(
        self,
        texts: list[str],
        ids: list[str],
        metadatas: list[Dict[str, Any]],
    ) -> None:
        """
        Embed and upsert a batch.
        """
        embeddings = self.embedder.embed_documents(texts)

        self.vector_store.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )