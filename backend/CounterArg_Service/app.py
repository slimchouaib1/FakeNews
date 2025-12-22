from fastapi import FastAPI
from pydantic import BaseModel

from .generator import CounterArgGenerator

app = FastAPI()

# ✅ Important : ne pas casser l'import si torch n'est pas installé
gen = None
GEN_AVAILABLE = False

try:
    gen = CounterArgGenerator()
    GEN_AVAILABLE = True
except RuntimeError:
    # torch/transformers non disponibles en CI
    GEN_AVAILABLE = False


class PredictRequest(BaseModel):
    claim: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    # ✅ Toujours répondre 200, même si torch absent
    if not GEN_AVAILABLE or gen is None:
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
