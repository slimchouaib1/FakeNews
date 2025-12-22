from __future__ import annotations

import os
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel

from .config import settings
from .generator import build_prompt

app = FastAPI(title="Counter-Argument Service")


class GenerateRequest(BaseModel):
    claim: str
    evidence: str | None = ""


class GenerateResponse(BaseModel):
    counterArgument: str
    device: str


class DummyGenerator:
    """Utilisé en CI (pas de dépendances lourdes)."""

    def generate(self, claim: str, evidence: str = ""):
        prompt = build_prompt(claim, evidence)
        # réponse simple et déterministe pour les tests
        return {
            "counterArgument": f"[CI MODE] Contre-argument basé sur le prompt: {prompt[:120]}...",
            "device": "cpu",
        }


@lru_cache
def get_generator():
    # Si on est en CI, on évite de charger un modèle lourd
    if os.getenv("CI", "").lower() == "true" or os.getenv("SKIP_MODEL_LOAD", "").lower() == "true":
        return DummyGenerator()

    # Mode normal: vrai modèle
    from .generator import CounterArgGenerator
    return CounterArgGenerator()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": settings.model_name,
        "device": settings.DEVICE,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    gen = get_generator()

    # DummyGenerator retourne déjà le bon format
    if isinstance(gen, DummyGenerator):
        out = gen.generate(req.claim, req.evidence or "")
        return GenerateResponse(counterArgument=out["counterArgument"], device=out["device"])

    result = gen.generate(req.claim, req.evidence or "")
    return GenerateResponse(counterArgument=result.counter_argument, device=result.used_device)
