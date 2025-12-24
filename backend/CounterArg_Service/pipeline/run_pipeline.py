from __future__ import annotations

from backend.CounterArg_Service.pipeline.prepare_data import main as prepare_main
from backend.CounterArg_Service.pipeline.eval_generation import main as eval_main

# (optionnel) si train existe, on le rend safe
try:
    from backend.CounterArg_Service.pipeline.train import main as train_main
except Exception:
    train_main = None


def main():
    # 1) Préparer dataset
    prepare_main()  # génère claims.jsonl

    # 2) (optionnel) train si dispo
    if train_main is not None:
        train_main()

    # 3) Évaluer
    eval_main()


if __name__ == "__main__":
    main()
