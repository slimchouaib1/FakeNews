from __future__ import annotations

from pydantic import BaseModel, ConfigDict
import os


class Settings(BaseModel):
    # évite le warning "protected namespace model_"
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = os.getenv("MODEL_NAME", "google/flan-t5-base")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))

    # on garde un champ "device" (minuscule)
    device: str = os.getenv("DEVICE", "cpu")


settings = Settings()
