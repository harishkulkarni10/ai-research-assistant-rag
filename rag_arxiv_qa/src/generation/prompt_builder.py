from typing import List, Dict, Any
import tiktoken

    
class PromptBuilder:
    """
    - Injects metadata
    - Enforces context token budget
    - Forces structured JSON output
    """

    def __init__(self, config: Dict[str, Any]):
        self.max_context_tokens = config["generation"].get("max_context_tokens", 1500)
        self.model_name = config["generation"]["model"]

        # Tokenizer for budgeting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _build_context_block(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Add chunks until token budget reached.
        Drops overflow safely.
        """

        context_sections = []
        total_tokens = 0

        for idx, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "unknown")
            position = metadata.get("position", "NA")

            block = (
                f"[{idx}] "
                f"(Source: {source}, Position: {position})\n"
                f"{chunk['text']}\n\n"
            )

            block_tokens = self._count_tokens(block)

            if total_tokens + block_tokens > self.max_context_tokens:
                break

            context_sections.append(block)
            total_tokens += block_tokens

        return "".join(context_sections)

    def build(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Build final prompt.
        """

        context = self._build_context_block(chunks)

        system_instruction = """
        You are a research assistant.
        Answer strictly using the provided context. 
        If the answer is not in the context, say "Insufficient information."
 
        Return ONLY valid JSON in this exact format:

        {
            "answer": "...",
            "citations": [1, 2],
            "confidence_score": 0.0-1.0
        }
        
        """

        user_prompt = f"""
        Context:
        {context}

        Question:
        {query}
        """

        return system_instruction.strip() + "\n\n" + user_prompt.strip()