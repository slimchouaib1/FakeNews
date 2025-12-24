from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict


DEFAULT_SAMPLES: List[Dict[str, str]] = [
    {"claim": "The earth is flat."},
    {"claim": "Vaccines cause autism."},
    {"claim": "Climate change is a hoax."},
    {"claim": "Humans never landed on the Moon."},
    {"claim": "Drinking bleach cures diseases."},
    {"claim": "5G networks spread viruses."},
    {"claim": "The sun revolves around the Earth."},
    {"claim": "All news on the internet is true."},
    {"claim": "Dinosaurs lived with humans."},
    {"claim": "Antibiotics kill viruses."},
    {"claim": "A single food can detox the body completely."},
    {"claim": "The Great Wall of China is visible from space with naked eye."},
    {"claim": "Lightning never strikes the same place twice."},
    {"claim": "Goldfish have a 3-second memory."},
    {"claim": "Sugar makes children hyperactive."},
    {"claim": "You only use 10% of your brain."},
    {"claim": "Cracking knuckles causes arthritis."},
    {"claim": "Cold weather causes colds."},
    {"claim": "Earth has only one continent."},
    {"claim": "Photosynthesis happens only at night."},
]


def write_dataset(out_path: Path, samples: List[Dict[str, str]]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    # dataset minimal local, versionnable, reproductible
    out_path = Path("backend/CounterArg_Service/pipeline/data/claims.jsonl")
    write_dataset(out_path, DEFAULT_SAMPLES)
    print(f"✅ Dataset généré: {out_path} ({len(DEFAULT_SAMPLES)} lignes)")


if __name__ == "__main__":
    main()
