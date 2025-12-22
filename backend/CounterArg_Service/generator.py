from typing import Dict, Any
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import settings, load_best_config


def load_prompt_template() -> str:
    with open(settings.PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(claim: str, context: str = "") -> str:
    template = load_prompt_template()
    # If context is provided (RAG on), inject it before the claim in a safe way
    if context:
        return template.replace("{claim}", f"{claim}\n\nContext:\n{context}")
    return template.format(claim=claim)


def format_ok(text: str) -> bool:
    has_bullets = ("- " in text) or ("•" in text)
    long_enough = len(text.split()) >= 25
    return has_bullets and long_enough


class CounterArgGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME, use_fast=True)

        kwargs = {
            "device_map": settings.DEVICE_MAP,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if settings.QUANT == "4bit":
            kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(settings.MODEL_NAME, **kwargs)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, claim: str, context: str = "") -> Dict[str, Any]:
        best = load_best_config()

        max_new_tokens = int(best.get("max_new_tokens", settings.MAX_NEW_TOKENS))
        temperature = float(best.get("temperature", settings.TEMPERATURE))
        top_p = float(best.get("top_p", settings.TOP_P))

        prompt = build_prompt(claim, context=context)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        t0 = time.time()
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        latency_ms = (time.time() - t0) * 1000

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        key = "Counter-argument:"
        if key in decoded:
            decoded = decoded.split(key, 1)[-1].strip()

        return {
            "counter_argument": decoded,
            "latency_ms": latency_ms,
            "format_ok": format_ok(decoded),
            "model": settings.MODEL_NAME,
            "prompt_version": settings.PROMPT_VERSION,
            "params": {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens},
        }
