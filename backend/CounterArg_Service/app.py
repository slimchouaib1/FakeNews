from __future__ import annotations

import os
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel

from .config import settings
from .generator import build_prompt


app = FastAPI(title="Counter-Argument Service")


class PredictRequest(BaseModel):
    claim: str


class PredictResponse(BaseModel):
    counterArgument: str


class GenerateRequest(BaseModel):
    claim: str
    evidence: str | None = ""


class GenerateResponse(BaseModel):
    counterArgument: str
    device: str


class DummyGenerator:
    def generate(self, claim: str, evidence: str = ""):
        prompt = build_prompt(claim, evidence)
        return {
            "counterArgument": f"[CI MODE] Contre-argument basé sur: {prompt[:120]}...",
            "device": "cpu",
        }


@lru_cache
def get_generator():
    if os.getenv("CI", "").lower() == "true" or os.getenv("SKIP_MODEL_LOAD", "").lower() == "true":
        return DummyGenerator()

    from .generator import CounterArgGenerator
    return CounterArgGenerator()


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.model_name, "device": settings.DEVICE}


# ✅ endpoint attendu par les tests
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    gen = get_generator()
    out = gen.generate(req.claim, "")
    return PredictResponse(counterArgument=out["counterArgument"])


# endpoint optionnel (si tu veux garder /generate)
@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    gen = get_generator()
    out = gen.generate(req.claim, req.evidence or "")
    return GenerateResponse(counterArgument=out["counterArgument"], device=out["device"])
