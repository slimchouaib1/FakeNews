from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

from backend.CounterArg_Service.generator import CounterArgGenerator, build_prompt


def load_claims(path: Path) -> List[str]:
    claims: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            c = (obj.get("claim") or "").strip()
            if c:
                claims.append(c)
    return claims


def safe_generate(gen: CounterArgGenerator, claim: str) -> Tuple[str, Dict[str, Any]]:
    """
    Génère une réponse. Si torch n'est pas dispo, on renvoie un fallback
    pour que la pipeline reste exécutable (utile en CI).
    """
    # si ton generator lève encore RuntimeError quand torch absent,
    # cette fonction protège la pipeline.
    try:
        res = gen.generate(claim)
        # res peut être string OU dataclass/objet
        if isinstance(res, str):
            return res, {"used_device": "cpu", "model_loaded": True}
        # dataclass ?
        if hasattr(res, "__dict__"):
            d = dict(res.__dict__)
            return d.get("counter_argument", str(res)), d
        return str(res), {"used_device": "cpu", "model_loaded": True}
    except Exception as e:
        return (
            "(Mode CI) Génération indisponible (torch/transformers non installés).",
            {"used_device": "cpu", "model_loaded": False, "error": repr(e)},
        )


def eval_predictions(preds: List[str], latencies_ms: List[float]) -> Dict[str, float]:
    non_empty = [p for p in preds if isinstance(p, str) and p.strip()]
    avg_len = sum(len(p) for p in non_empty) / max(1, len(non_empty))
    non_empty_rate = len(non_empty) / max(1, len(preds))
    avg_latency = sum(latencies_ms) / max(1, len(latencies_ms))
    return {
        "avg_length_chars": float(avg_len),
        "non_empty_rate": float(non_empty_rate),
        "avg_latency_ms": float(avg_latency),
    }


def main() -> None:
    data_path = Path("backend/CounterArg_Service/pipeline/data/claims.jsonl")
    out_path = Path("backend/CounterArg_Service/pipeline/outputs/predictions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    claims = load_claims(data_path)
    if not claims:
        raise RuntimeError(f"Aucun claim trouvé dans {data_path}")

    gen = CounterArgGenerator()

    preds: List[str] = []
    lat_ms: List[float] = []

    with out_path.open("w", encoding="utf-8") as f:
        for claim in claims:
            t0 = time.perf_counter()
            text, meta = safe_generate(gen, claim)
            t1 = time.perf_counter()

            latency = (t1 - t0) * 1000.0
            lat_ms.append(latency)
            preds.append(text)

            row = {
                "claim": claim,
                "prompt_preview": build_prompt(claim)[:200],
                "prediction": text,
                "latency_ms": latency,
                "meta": meta,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = eval_predictions(preds, lat_ms)
    print("✅ Évaluation terminée.")
    print("📊 Metrics:", metrics)
    print(f"📁 Outputs: {out_path}")


if __name__ == "__main__":
    main()
