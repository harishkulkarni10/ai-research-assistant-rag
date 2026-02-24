# LLM provider implementations for different backends
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import logging
import requests

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass


class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", **kwargs):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 512)
        
    def generate(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?")
            raise ConnectionError(f"Ollama not available at {self.base_url}. Please start Ollama.")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise


class HuggingFaceProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.base_url = "https://api-inference.huggingface.co/models"
        
        if not self.api_key:
            logger.warning("No HuggingFace API key provided. Using public endpoint (may be rate-limited).")
    
    def generate(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}/{self.model_name}"
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                "return_full_text": False,
            }
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict):
                return result.get("generated_text", "")
            else:
                return str(result)
        except requests.exceptions.HTTPError as e:
            if response.status_code == 503:
                logger.warning("Model is loading. This may take a minute on first request.")
                import time
                time.sleep(10)
                return self.generate(prompt, **kwargs)
            logger.error(f"HuggingFace API error: {e}")
            raise
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise


class VLLMProvider(LLMProvider):
    def __init__(self, model_name: str, base_url: str = "http://localhost:8000", **kwargs):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 512)
        
    def generate(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}/v1/completions"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stop": None,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["text"].strip()
            else:
                raise ValueError("Unexpected vLLM response format")
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to vLLM server at {self.base_url}. Is vLLM running?")
            raise ConnectionError(f"vLLM server not available at {self.base_url}. Please start vLLM server.")
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            raise


class TGIProvider(LLMProvider):
    def __init__(self, model_name: str, base_url: str = "http://localhost:8080", **kwargs):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 512)
        
    def generate(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}/generate"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                "return_full_text": False,
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict):
                return result.get("generated_text", "")
            else:
                return str(result)
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to TGI server at {self.base_url}. Is TGI running?")
            raise ConnectionError(f"TGI server not available at {self.base_url}. Please start TGI server.")
        except Exception as e:
            logger.error(f"TGI generation error: {e}")
            raise


class TransformersProvider(LLMProvider):
    def __init__(self, model_name: str, **kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model_name = model_name
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading local model: {self.model_name} on {self.device}")
        
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
    
    def generate(self, prompt: str, **kwargs) -> str:
        import torch
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=kwargs.get("temperature", self.temperature) > 0,
            )
        
        raw_output = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        
        if raw_output.startswith(prompt):
            raw_output = raw_output[len(prompt):].strip()
        
        return raw_output


def get_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    gen_config = config.get("generation", {})
    provider = gen_config.get("provider", "ollama").lower()
    model_name = gen_config.get("model", "llama3.2:1b")
    
    provider_kwargs = {
        "temperature": gen_config.get("temperature", 0.2),
        "max_tokens": gen_config.get("max_tokens", 512),
    }
    
    if provider == "ollama":
        base_url = gen_config.get("base_url", "http://localhost:11434")
        return OllamaProvider(model_name, base_url=base_url, **provider_kwargs)
    
    elif provider == "huggingface":
        api_key = gen_config.get("api_key") or config.get("secrets", {}).get("hf_api_key")
        if not api_key:
            import os
            api_key = os.getenv("HUGGINGFACE_API_KEY")
        return HuggingFaceProvider(model_name, api_key=api_key, **provider_kwargs)
    
    elif provider == "vllm":
        base_url = gen_config.get("base_url", "http://localhost:8000")
        return VLLMProvider(model_name, base_url=base_url, **provider_kwargs)
    
    elif provider == "tgi":
        base_url = gen_config.get("base_url", "http://localhost:8080")
        return TGIProvider(model_name, base_url=base_url, **provider_kwargs)
    
    elif provider == "transformers":
        return TransformersProvider(model_name, **provider_kwargs)
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: ollama, huggingface, vllm, tgi, transformers")
