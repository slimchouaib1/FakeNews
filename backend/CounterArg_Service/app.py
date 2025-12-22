from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from config import settings
from generator import CounterArgGenerator
from monitoring.logger import log_request

app = FastAPI(title="Counter-Argumentation Service", version="1.0")

# Lazy-loaded generator (avoid model load at import time)
_gen: Optional[CounterArgGenerator] = None


def get_generator() -> CounterArgGenerator:
    global _gen
    if _gen is None:
        _gen = CounterArgGenerator()
    return _gen


class GenerateRequest(BaseModel):
    text: str = Field(..., description="The claim / fake-news text")
    label: Optional[str] = Field(None, description="Optional: Fake/Real from classifier")
    confidence: Optional[float] = Field(None, description="Optional: classifier confidence")


class GenerateResponse(BaseModel):
    counter_argument: str
    metadata: Dict[str, Any]


@app.post("/generate-counter", response_model=GenerateResponse)
def generate_counter(req: GenerateRequest):
    # Optional guard: only generate if fake
    if req.label is not None and req.label.lower() != "fake":
        return GenerateResponse(
            counter_argument="No counter-argument generated because the content was not labeled as FAKE.",
            metadata={"label": req.label, "confidence": req.confidence},
        )

    generator = get_generator()
    result = generator.generate(req.text)

    log_request(
        endpoint="/generate-counter",
        input_len=len(req.text),
        latency_ms=result["latency_ms"],
        model=result["model"],
        prompt_version=result["prompt_version"],
        params=result["params"],
        format_ok=result["format_ok"],
    )

    return GenerateResponse(
        counter_argument=result["counter_argument"],
        metadata={
            "label": req.label,
            "confidence": req.confidence,
            "model": result["model"],
            "prompt_version": result["prompt_version"],
            "latency_ms": result["latency_ms"],
            "format_ok": result["format_ok"],
            "params": result["params"],
        },
    )


@app.get("/health")
def health():
    # IMPORTANT: should not load the model
    return {
        "ok": True,
        "model": settings.MODEL_NAME,
        "prompt_version": settings.PROMPT_VERSION,
