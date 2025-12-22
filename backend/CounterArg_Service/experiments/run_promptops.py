import json
import mlflow
from generator import CounterArgGenerator
from config import settings

CLAIMS = [
    "Vaccines are always dangerous and should be avoided.",
    "Climate change is a hoax created to control people.",
    "Drinking lemon water cures all diseases.",
]

GRID = [
    {"temperature": 0.2, "top_p": 0.9, "max_new_tokens": 200},
    {"temperature": 0.4, "top_p": 0.9, "max_new_tokens": 220},
    {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 220},
]

def local_format_ok(text: str) -> int:
    return 1 if (("- " in text or "•" in text) and len(text.split()) >= 25) else 0

def main():
    mlflow.set_experiment("counter_argument_promptops")
    gen = CounterArgGenerator()

    records = []
    for cfg in GRID:
        with mlflow.start_run():
            mlflow.log_params({"model_name": settings.MODEL_NAME, "prompt_version": settings.PROMPT_VERSION, **cfg})

            latencies, oks, out_lens = [], [], []
            for claim in CLAIMS:
                res = gen.generate(claim)
                latencies.append(res["latency_ms"])
                ok = local_format_ok(res["counter_argument"])
                oks.append(ok)
                out_lens.append(len(res["counter_argument"].split()))

                records.append({
                    "cfg": cfg,
                    "claim": claim,
                    "counter_argument": res["counter_argument"],
                    "latency_ms": res["latency_ms"],
                    "format_ok": ok,
                })

            mlflow.log_metric("avg_latency_ms", sum(latencies)/len(latencies))
            mlflow.log_metric("format_ok_rate", sum(oks)/len(oks))
            mlflow.log_metric("avg_output_len", sum(out_lens)/len(out_lens))

            mlflow.log_artifact(settings.PROMPT_PATH)

    # Save artifacts for later pipeline selection
    import os
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/run_outputs.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Saved artifacts/run_outputs.jsonl")
    print("Run MLflow UI with: mlflow ui")

if __name__ == "__main__":
    main()
