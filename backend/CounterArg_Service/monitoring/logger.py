import json
import time
import uuid

def log_request(endpoint: str, input_len: int, latency_ms: float, model: str,
                prompt_version: str, params: dict, format_ok: bool):
    record = {
        "ts": time.time(),
        "request_id": str(uuid.uuid4()),
        "endpoint": endpoint,
        "input_len": input_len,
        "latency_ms": latency_ms,
        "model": model,
        "prompt_version": prompt_version,
        "params": params,
        "format_ok": format_ok,
    }
    print(json.dumps(record, ensure_ascii=False))
