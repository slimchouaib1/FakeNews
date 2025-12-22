from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = "google/flan-t5-base"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    use_gpu: bool = False
    device: str = "cpu"


settings = Settings()
