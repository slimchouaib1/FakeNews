from pathlib import Path

LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "artifacts" / "counterarg_model"
model_source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else settings.model_name

self._tokenizer = AutoTokenizer.from_pretrained(model_source)
self._model = AutoModelForSeq2SeqLM.from_pretrained(model_source).to(self.device)

