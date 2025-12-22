# backend/CounterArg_Service/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from .generator import CounterArgGenerator

app = FastAPI(title="CounterArg Service", version="1.0.0")

gen = CounterArgGenerator()

class PredictRequest(BaseModel):
    claim: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    result = gen.generate(req.claim)
    return {
        "claim": req.claim,
        "counterArgument": result["counterArgument"],
        "latency": result["latency"],
        "device": result["device"]
    }
