# backend/CounterArg_Service/pipeline/train.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from backend.CounterArg_Service.config import settings
from backend.CounterArg_Service.generator import build_prompt


ROOT = Path(__file__).resolve().parents[2]  # backend/CounterArg_Service
PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = PIPELINE_DIR / "data"
DEFAULT_DATA_PATH = DATA_DIR / "claims.jsonl"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_OUT_DIR = ARTIFACTS_DIR / "counterarg_model"


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_xy(row: Dict) -> Tuple[str, str]:
    claim = (row.get("claim") or "").strip()

    # target field (support multiple names)
    y = (
        row.get("counter_argument")
        or row.get("counterArgument")
        or row.get("target")
        or ""
    ).strip()

    if not claim:
        raise ValueError("Ligne invalide: 'claim' est vide.")
    if not y:
        raise ValueError(
            "Dataset invalide: target manquante. "
            "Ajoute 'counter_argument' (ou 'counterArgument' ou 'target') dans ton jsonl."
        )
    return claim, y


class JsonlSeq2SeqDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer: AutoTokenizer, max_src: int = 384, max_tgt: int = 256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        claim, target = _extract_xy(row)

        prompt = build_prompt(claim, row.get("evidence", "") or "")

        model_inputs = self.tokenizer(
            prompt,
            max_length=self.max_src,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target,
                max_length=self.max_tgt,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )

        item = {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }
        return item


def train_main(
    data_path: Optional[str] = None,
    experiment_name: str = "counterarg-train",
) -> Path:
    # --- paths ---
    path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    # --- device ---
    wanted = getattr(settings, "DEVICE", "cpu")
    device = "cuda" if (wanted.lower() == "cuda" and torch.cuda.is_available()) else "cpu"

    # --- load data ---
    rows = _read_jsonl(path)
    if len(rows) < 4:
        raise ValueError("Dataset trop petit. Mets au moins 4 lignes pour split train/val.")

    # split simple
    n = len(rows)
    val_size = max(1, int(0.2 * n))
    train_rows = rows[:-val_size]
    val_rows = rows[-val_size:]

    # --- model/tokenizer ---
    model_name = settings.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    train_ds = JsonlSeq2SeqDataset(train_rows, tokenizer)
    val_ds = JsonlSeq2SeqDataset(val_rows, tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # --- training args (CPU friendly) ---
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_OUT_DIR),
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,

        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # stabilise sur CPU
        learning_rate=5e-5,

        predict_with_generate=False,
        fp16=False,
        bf16=False,

        report_to=[],  # on gère MLflow manuellement
    )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"train-{model_name.replace('/', '_')}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("device", device)
        mlflow.log_param("train_size", len(train_rows))
        mlflow.log_param("val_size", len(val_rows))
        mlflow.log_param("epochs", args.num_train_epochs)
        mlflow.log_param("lr", args.learning_rate)
        mlflow.log_param("grad_accum", args.gradient_accumulation_steps)

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            tokenizer=tokenizer,
        )

        trainer.train()

        # Eval loss
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        # save + log artifacts
        trainer.save_model(str(MODEL_OUT_DIR))
        tokenizer.save_pretrained(str(MODEL_OUT_DIR))

        mlflow.log_artifacts(str(MODEL_OUT_DIR), artifact_path="model")

    print(f"✅ Modèle entraîné sauvegardé dans: {MODEL_OUT_DIR}")
    return MODEL_OUT_DIR


if __name__ == "__main__":
    train_main()
