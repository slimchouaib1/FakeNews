from fastapi import FastAPI
from pydantic import BaseModel

from .generator import CounterArgGenerator

app = FastAPI()

gen = CounterArgGenerator()


class PredictRequest(BaseModel):
    claim: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    return {
        "counterArgument": gen.generate(req.claim)
    }
