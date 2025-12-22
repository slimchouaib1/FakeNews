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
            f"Claim:\n{claim}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Contre-argument (court, structuré, sans halluciner):"
        )
    return (
        "Tu es un assistant qui génère un contre-argument clair et factuel.\n\n"
        f"Claim:\n{claim}\n\n"
        "Contre-argument (court, structuré, sans halluciner):"
    )


@dataclass
class GenerationResult:
    counter_argument: str
    used_device: str = "cpu"


class CounterArgGenerator:
    """
    Charge le modèle uniquement quand on instancie CounterArgGenerator().
    En CI, on peut éviter d'instancier ce composant (voir app.py).
    """

    def __init__(self) -> None:
        self.device = "cpu"
        self._tokenizer = None
        self._model = None

        # Lazy import (évite de casser les tests si torch n'est pas installé)
        try:
            import torch  # noqa
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa
        except Exception as e:
            raise RuntimeError(
                "Dépendances ML non disponibles (torch/transformers). "
                "Installe les requirements 'full' pour activer la génération."
            ) from e

        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        wanted = settings.DEVICE.lower()
        self.device = "cuda" if (wanted == "cuda" and torch.cuda.is_available()) else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(settings.model_name).to(self.device)
        self._model.eval()

    def generate(self, claim: str, evidence: str = "") -> GenerationResult:
        prompt = build_prompt(claim, evidence)

        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                do_sample=True,
            )

        text = self._tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return GenerationResult(counter_argument=text, used_device=self.device)
