
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    Production-ready config loader with deep merge, .env support, and basic validation.
    """
    def __init__(self, env: str = None):
        # Find config folder (3 levels up from config.py)
        base_path = Path(__file__).resolve().parents[3] / "config"

        # Load base config
        base_file = base_path / "config.yaml"
        if not base_file.exists():
            raise FileNotFoundError(f"Base config not found: {base_file}")
        with open(base_file, "r") as f:
            self.config = yaml.safe_load(f) or {}

        # Determine environment
        env = env or os.getenv("APP_ENV") or self.config.get("project", {}).get("environment", "dev")
        env_file = base_path / f"{env}.yaml"

        # Load and merge env override
        if env_file.exists():
            with open(env_file, "r") as f:
                env_config = yaml.safe_load(f) or {}
                self._deep_update(self.config, env_config)
        else:
            print(f"Warning: No override file found for env '{env}'")

        # Override any value from environment variables (RAG_ prefix)
        for key, value in os.environ.items():
            if key.startswith("RAG_"):
                parts = key[4:].lower().split("_")
                current = self.config
                for p in parts[:-1]:
                    current = current.setdefault(p, {})
                current[parts[-1]] = value

        # Basic validation (add more as needed)
        required_keys = ["embeddings.model", "vector_db.provider", "generation.model"]
        for key in required_keys:
            parts = key.split(".")
            val = self.config
            for p in parts:
                val = val.get(p)
                if val is None:
                    raise ValueError(f"Missing required config key: {key}")

    def _deep_update(self, base: Dict, updates: Dict):
        """Recursively merge updates into base."""
        for k, v in updates.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v

    def get(self) -> Dict[str, Any]:
        return self.config


def load_config(env: str = None) -> Dict[str, Any]:
    """Global access point."""
    return Config(env).get()