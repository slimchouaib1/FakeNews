from fastapi import FastAPI
from pydantic import BaseModel

from .generator import CounterArgGenerator

app = FastAPI()

gen = None



class PredictRequest(BaseModel):
    claim: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    global gen

    # Lazy load: on instancie seulement quand /predict est appelé
    if gen is None:
        try:
            gen = CounterArgGenerator()
        except Exception:
            # ✅ Toujours répondre 200 même si torch absent en CI
            return {
                "counterArgument": "(Mode CI) Dépendances ML (torch/transformers) non installées. "
                                   "Installe la version full pour activer la génération.",
                "used_device": "cpu",
                "model_loaded": False
            }

    res = gen.generate(req.claim)

    # si res est string ou objet (selon ton generator), on gère les 2 cas
    if isinstance(res, str):
        return {"counterArgument": res}

    return {
        "counterArgument": getattr(res, "counter_argument", str(res)),
        "used_device": getattr(res, "used_device", "cpu"),
        "model_loaded": getattr(res, "model_loaded", True),
    }
