import os
import json

class Settings(BaseModel):
    MODEL_NAME: str = os.getenv("MODEL_NAME", "google/flan-t5-base")
    DEVICE: str = os.getenv("DEVICE", "cpu")  # "cpu" or "cuda"
    USE_QUANT: bool = os.getenv("USE_QUANT", "false").lower() == "true"
    BEST_CONFIG_PATH: str = os.getenv(
        "BEST_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "best_config.json")
    )

settings = Settings()

def load_best_config(path: str | None = None) -> dict:
    path = path or settings.BEST_CONFIG_PATH
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # default safe config
    return {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 200}
