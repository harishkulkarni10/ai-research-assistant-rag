from typing import Dict, Iterable
import hashlib

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker: 
    """
    Converts a single document into token-aware chunks
    """
    def __init__(self, config: Dict):
        chunk_cfg = config['chunking']

        self.chunk_size = chunk_cfg['chunk_size_tokens']
        self.chunk_overlap = chunk_cfg['chunk_overlap_tokens']
        self.min_chunk_tokens = chunk_cfg['min_chunk_tokens']

        tokenizer_name = chunk_cfg.get('tokenizer', 'cl100k_base')
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True,
            strip_whitespace=True,
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _make_chunk_id(self, doc_id: str, chunk_text: str) -> str:
        payload = f"{doc_id}:{chunk_text}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()


    def chunk_document(
        self, 
        *, 
        doc_id: str, 
        text: str, 
        base_metadata: Dict,
    ) -> Iterable[Dict]:
        raw_chunks = self.splitter.split_text(text)

        for position, chunk_text in enumerate(raw_chunks):
            token_count = self._count_tokens(chunk_text)

            if token_count < self.min_chunk_tokens:
                continue

            yield {
                "chunk_id": self._make_chunk_id(doc_id, chunk_text),
                "doc_id": doc_id,
                "text": chunk_text,
                "metadata": {
                    **base_metadata,
                    "position": position,
                    "token_count": token_count,
                }
            }
