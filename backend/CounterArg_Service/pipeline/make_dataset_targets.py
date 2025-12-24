# backend/CounterArg_Service/pipeline/make_dataset_targets.py
import json
from pathlib import Path

IN_PATH = Path("backend/CounterArg_Service/pipeline/data/claims.jsonl")
OUT_PATH = Path("backend/CounterArg_Service/pipeline/data/claims_with_targets.jsonl")

def simple_counter_argument(claim: str) -> str:
    claim = (claim or "").strip()
    return (
        "Contre-argument (baseline):\n"
        f"- La phrase « {claim} » nécessite des preuves vérifiables.\n"
        "- Sans sources fiables et consensus scientifique, on ne peut pas la considérer comme un fait.\n"
        "- Il faut consulter des références reconnues (études, institutions, experts)."
    )

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable: {IN_PATH}")

    n = 0
    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            claim = row.get("claim", "").strip()
            if not claim:
                continue

            # si target existe déjà on la garde
            if not row.get("counter_argument"):
                row["counter_argument"] = simple_counter_argument(claim)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"✅ Dataset prêt: {OUT_PATH} ({n} lignes)")

if __name__ == "__main__":
    main()
