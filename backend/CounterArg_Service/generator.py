# backend/CounterArg_Service/generator.py
from __future__ import annotations
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .config import settings, load_best_config

def build_prompt(claim: str) -> str:
    claim = (claim or "").strip()
    return (
        "You are a fact-checking assistant.\n"
        "Write a concise counter-argument to this claim.\n\n"
        f"CLAIM: {claim}\n"
        "COUNTER-ARGUMENT:"
    )

class CounterArgGenerator:
    """
    CPU-safe generator.
    - No quantization in CI (no GPU).
    - Lazy load possible.
    """
    def __init__(self):
        self.cfg = load_best_config()
        self.device = "cuda" if (settings.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"

        # ✅ IMPORTANT: do NOT try 4bit/8bit in CI CPU
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, claim: str) -> dict:
        prompt = build_prompt(claim)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.get("max_new_tokens", 200)),
                do_sample=True,
                temperature=float(self.cfg.get("temperature", 0.7)),
                top_p=float(self.cfg.get("top_p", 0.9)),
            )
        latency = time.time() - t0

        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        return {
            "counterArgument": text,
            "latency": latency,
            "device": self.device
        }
