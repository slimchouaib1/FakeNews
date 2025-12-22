from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import settings


def build_prompt(claim: str, evidence: str = "") -> str:
    claim = (claim or "").strip()
    evidence = (evidence or "").strip()

    if evidence:
        return (
            "Tu es un assistant qui génère un contre-argument clair et factuel.\n\n"
            f"CLAIM:\n{claim}\n\n"
            f"EVIDENCE:\n{evidence}\n\n"
            "COUNTER-ARGUMENT:"
        )

    return (
        "Tu es un assistant qui génère un contre-argument clair et factuel.\n\n"
        f"CLAIM:\n{claim}\n\n"
        "COUNTER-ARGUMENT:"
    )


@dataclass
class GenerationResult:
    counter_argument: str
    used_device: str = "cpu"
    model_loaded: bool = False


class CounterArgGenerator:
    """
    Générateur robuste :
    - En local (full deps), il charge le modèle.
    - En CI (sans torch), il ne casse pas les imports/tests.
    """

    def __init__(self) -> None:
        self.device = "cpu"
        self._tokenizer = None
        self._model = None
        self.available = False  # torch / transformers disponibles ?

        try:
            import torch  # noqa
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa
        except Exception:
            # ⚠️ Pas d'exception ici : CI doit passer
            self.available = False
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        wanted = (settings.device or "cpu").lower()
        self.device = "cuda" if (wanted == "cuda" and torch.cuda.is_available()) else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.model_name
        ).to(self.device)
        self._model.eval()

        self.available = True

    def generate(self, claim: str, evidence: str = "") -> GenerationResult:
        prompt = build_prompt(claim, evidence)

        # 🔹 Mode CI / environnement minimal
        if not self.available:
            return GenerationResult(
                counter_argument=(
                    "(Mode CI) Dépendances ML non installées. "
                    "Installe la version full pour activer la génération."
                ),
                used_device="cpu",
                model_loaded=False,
            )

        import torch

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).to(self.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                do_sample=True,
            )

        text = self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

        return GenerationResult(
            counter_argument=text,
            used_device=self.device,
            model_loaded=True,
        )
