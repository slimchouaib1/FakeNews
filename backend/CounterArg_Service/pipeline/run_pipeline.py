from __future__ import annotations

import os
from pathlib import Path

import mlflow

from backend.CounterArg_Service.config import settings


from backend.CounterArg_Service.pipeline.prepare_data import main as prepare_main
from backend.CounterArg_Service.pipeline.train import train_main
from backend.CounterArg_Service.pipeline.eval_generation import main as eval_main

def main():
    prepare_main()          # génère claims.jsonl
    train_main()            # fine-tune le modèle (MLflow run)
    eval_main()             # évalue et log metrics (si ton eval le fait)
    data_path = Path("backend/CounterArg_Service/pipeline/data/claims.jsonl")
    out_path = Path("backend/CounterArg_Service/pipeline/outputs/predictions.jsonl")

    # MLflow setup
    mlflow.set_experiment("counter-argument-generation")

    with mlflow.start_run(run_name="eval-only") as run:
        # ---- Params (depuis settings) ----
        mlflow.log_param("model_name", getattr(settings, "model_name", "unknown"))
        mlflow.log_param("max_new_tokens", getattr(settings, "max_new_tokens", None))
        mlflow.log_param("temperature", getattr(settings, "temperature", None))
        mlflow.log_param("top_p", getattr(settings, "top_p", None))
        mlflow.log_param("device", getattr(settings, "device", "cpu"))
        mlflow.log_param("ci", os.getenv("CI", "false"))

        # ---- Step 1: prepare dataset si absent ----
        if not data_path.exists():
            from backend.CounterArg_Service.pipeline.prepare_data import main as prepare_main
            prepare_main()

        # ---- Step 2: eval generation ----
        from backend.CounterArg_Service.pipeline.eval_generation import main as eval_main
        eval_main()

        # ---- Log artefacts ----
        if data_path.exists():
            mlflow.log_artifact(str(data_path))
        if out_path.exists():
            mlflow.log_artifact(str(out_path))

        # ---- Log metrics simples (re-lire outputs) ----
        # On recalcule vite depuis le fichier predictions
        import json

        preds = []
        lat = []
        model_loaded_flags = []

        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                preds.append((obj.get("prediction") or "").strip())
                lat.append(float(obj.get("latency_ms") or 0.0))
                meta = obj.get("meta") or {}
                model_loaded_flags.append(bool(meta.get("model_loaded", False)))

        non_empty = [p for p in preds if p]
        avg_len = sum(len(p) for p in non_empty) / max(1, len(non_empty))
        non_empty_rate = len(non_empty) / max(1, len(preds))
        avg_latency = sum(lat) / max(1, len(lat))
        loaded_rate = sum(1 for x in model_loaded_flags if x) / max(1, len(model_loaded_flags))

        mlflow.log_metric("avg_length_chars", float(avg_len))
        mlflow.log_metric("non_empty_rate", float(non_empty_rate))
        mlflow.log_metric("avg_latency_ms", float(avg_latency))
        mlflow.log_metric("model_loaded_rate", float(loaded_rate))

        print("✅ Run MLflow terminé :", run.info.run_id)


if __name__ == "__main__":
    main()
