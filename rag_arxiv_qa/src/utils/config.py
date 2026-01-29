import os
import yaml
from pathlib import Path

class Config:
    def __init__(self, env: str = None):
        base_path = Path(__file__).resolve().parents[3] / "config"
        with open(base_path / "config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        env = env or self.config["project"]["environment"]
        env_file = base_path / f"{env}.yaml"

        if env_file.exists():
            with open(env_file, "r") as f:
                env_config = yaml.safe_load(f)
                self._deep_update(self.config, env_config)

    def _deep_update(self, base: dict, updates: dict):
        for k, v in updates.items():
            if isinstance(v, dict) and k in base:
                self._deep_update(base[k], v)
            else:
                base[k] = v

    def get(self):
        return self.config


def load_config():
    env = os.getenv("APP_ENV", None)
    return Config(env).get()
