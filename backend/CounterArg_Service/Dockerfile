FROM python:3.12-slim

WORKDIR /app

# Petites libs utiles (optionnel)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 1) Copier requirements (meilleur cache)
COPY backend/CounterArg_Service/requirements.cpu.txt /app/requirements.cpu.txt

# 2) Installer deps (torch CPU-only)
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r /app/requirements.cpu.txt

# 3) Copier backend + artifacts (modèle fine-tuné)
COPY backend /app/backend

EXPOSE 8000

CMD ["uvicorn", "backend.CounterArg_Service.app:app", "--host", "0.0.0.0", "--port", "8000"]
