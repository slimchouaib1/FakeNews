from fastapi import FastAPI
from pydantic import BaseModel

from .generator import CounterArgGenerator


app = FastAPI(title="CounterArg Service", version="1.0.0")

gen = CounterArgGenerator()


class PredictRequest(BaseModel):
    # Le test envoie {"claim": "..."} donc on respecte ça
    claim: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Le test attend:
    - endpoint /predict
    - status_code 200
    """
    counter = gen.generate(req.claim)
    return {"counterArgument": counter}
