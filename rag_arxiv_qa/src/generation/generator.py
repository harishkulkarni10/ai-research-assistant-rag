from typing import List, Dict, Any
import json
import logging
from json_repair import repair_json

from rag_arxiv_qa.src.generation.prompt_builder import PromptBuilder
from rag_arxiv_qa.src.generation.llm_providers import get_llm_provider, LLMProvider


class Generator:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_tokens = config["generation"]["max_tokens"]
        self.temperature = config["generation"]["temperature"]

        logging.info(f"Initializing LLM provider: {config['generation'].get('provider', 'ollama')}")
        self.llm_provider: LLMProvider = get_llm_provider(config)
        self.prompt_builder = PromptBuilder(config)

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not chunks:
            return {
                "answer": "I am sorry, I do not have enough information to answer that.",
                "citations": [],
                "confidence_score": 0.0,
            }

        prompt = self.prompt_builder.build(query, chunks)
        raw_output = self.llm_provider.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        structured = self._safe_json_extract(raw_output)

        return structured

    def _safe_json_extract(self, text: str) -> Dict[str, Any]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            
            if start == -1 or end == 0:
                raise ValueError("No JSON found in output")
            
            json_str = text[start:end]
            from json_repair import repair_json
            fixed_json = repair_json(json_str)
            parsed = json.loads(fixed_json)

            return {
                "answer": parsed.get("answer", ""),
                "citations": parsed.get("citations", []),
                "confidence_score": parsed.get("confidence_score", 0.5),
            }
        except Exception as e:
            logging.warning(f"Failed to extract structured JSON from model output: {e}")
            logging.debug(f"Raw output: {text[:500]}...")
            return {
                "answer": text.strip(),
                "citations": [],
                "confidence_score": 0.3,
            }
