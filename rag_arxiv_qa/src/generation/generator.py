from typing import List, Dict, Any
import json
import logging
from json_repair import repair_json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_arxiv_qa.src.generation.prompt_builder import PromptBuilder


class Generator:
    """
    LLM generation pipeline.
        - Refusal if no context
        - Prompt construction
        - Model inference
        - Structured JSON output parsing
        - Confidence score estimation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["generation"]["model"]
        self.max_tokens = config["generation"]["max_tokens"]
        self.temperature = config["generation"]["temperature"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"Loading generation model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            ).to(self.device)

        self.prompt_builder = PromptBuilder(config)

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate structured answer from context.

        Returns:
            - Answer : string
            - Citations : list
            - Confidence score : float
        """

        # --------------------------------------------------------
        # Refusal if no context
        # --------------------------------------------------------
        if not chunks:
            return {
                "answer": "I am sorry, I do not have enough information to answer that.",
                "citations": [],
                "confidence_score": 0.0,
            }

        # --------------------------------------------------------
        # Build prompt
        # --------------------------------------------------------
        prompt = self.prompt_builder.build(query, chunks)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        # --------------------------------------------------------
        # Model inference
        # --------------------------------------------------------
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        raw_output = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # --------------------------------------------------------
        # Extract structured JSON
        # --------------------------------------------------------
        structured = self._safe_json_extract(raw_output)

        return structured

    def _safe_json_extract(self, text: str) -> Dict[str, Any]:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            json_str = text[start:end]

            parsed = json_repair.loads(json_str)

            return {
                "answer": parsed.get("answer", ""),
                "citations": parsed.get("citations", []),
                "confidence_score": parsed.get("confidence_score", 0.5),
            }
        except Exception:
            logging.warning("Failed to extract structured JSON from model output.")
            return {
                "answer": text.strip(),
                "citations": [],
                "confidence_score": 0.3,
            }
