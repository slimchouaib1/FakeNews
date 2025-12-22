from __future__ import annotations

from pydantic import BaseModel
from typing import Optional
import os
import json


class Settings(BaseModel):
    model_name: str = "google/flan-t5-base"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    use_gpu: bool = False
    device: str = "cpu"


def load_best_config(path: str = "best_config.json") -> dict:
    """
    Charge la meilleure configuration depuis un fichier JSON
    si le fichier existe.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Instance globale des paramètres
settings = Settings(**load_best_config())
