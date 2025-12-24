FROM python:3.12-slim

WORKDIR /app

# (Optionnel) petites libs système utiles pour certains wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copier d'abord les requirements (meilleur cache)
COPY backend/CounterArg_Service/requirements.cpu.txt /app/requirements.cpu.txt

# Installer torch CPU-only (évite CUDA + nvidia-*)
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r /app/requirements.cpu.txt

# Copier tout le repo
COPY . .

EXPOSE 8000

# Important: ton import uvicorn doit matcher ton package
CMD ["uvicorn", "backend.CounterArg_Service.app:app", "--host", "0.0.0.0", "--port", "8000"]
