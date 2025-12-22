import json
import os

ARTIFACTS_PATH = "artifacts/run_outputs.jsonl"
OUT_PATH = "configs/best_config.json"

def main():
    if not os.path.exists(ARTIFACTS_PATH):
        raise FileNotFoundError("Run experiments first: python experiments/run_promptops.py")

    stats = {}
    with open(ARTIFACTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = json.dumps(r["cfg"], sort_keys=True)
            stats.setdefault(key, {"cfg": r["cfg"], "n": 0, "ok": 0, "lat": 0.0})
            stats[key]["n"] += 1
            stats[key]["ok"] += int(r["format_ok"])
            stats[key]["lat"] += float(r["latency_ms"])

    scored = []
    for s in stats.values():
        ok_rate = s["ok"] / s["n"]
        avg_lat = s["lat"] / s["n"]
        scored.append((ok_rate, -avg_lat, s["cfg"]))

    scored.sort(reverse=True)
    best = scored[0][2]

    os.makedirs("configs", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("Best config saved to:", OUT_PATH)
    print(best)

if __name__ == "__main__":
    main()
