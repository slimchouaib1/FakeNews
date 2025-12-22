import os
import json

class Settings:
    # Pick a model you can run:
    # - "Qwen/Qwen2.5-1.5B-Instruct": Fast, lightweight, good for testing.
    # - "mistralai/Mistral-7B-Instruct-v0.3": Higher quality, heavier
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

    DEVICE_MAP: str = os.getenv("DEVICE_MAP", "auto")
    QUANT: str = os.getenv("QUANT", "4bit")

    # Generation defaults
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.25"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))

    # PromptOps (prompt versioning)
    PROMPT_VERSION: str = os.getenv("PROMPT_VERSION", "v1")
    PROMPT_PATH: str = os.getenv("PROMPT_PATH", "prompts/prompt_v1.txt")

    # Optional: Best config produced by pipeline (PromptOps tuning)
    BEST_CONFIG_PATH: str = os.getenv("BEST_CONFIG_PATH", "configs/best_config.json")

    # Optional RAG (keep it, but can be turned off)
    USE_RAG: bool = os.getenv("USE_RAG", "false").lower() == "true"
    KB_PATH: str = os.getenv("KB_PATH", "data/kb_docs.jsonl")
    TOP_K: int = int(os.getenv("TOP_K", "4"))

settings = Settings()

def load_best_config() -> dict:
    """
    Load best generation parameters selected by the MLOps pipeline.
    If missing/invalid -> return {} and fallback to defaults.
    """
    path = settings.BEST_CONFIG_PATH
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
