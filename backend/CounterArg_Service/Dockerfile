FROM python:3.12-slim
WORKDIR /app

# (optionnel) libs utiles
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Ici, le contexte = backend/, donc pas de "backend/" dans le chemin
COPY CounterArg_Service/requirements.cpu.txt /app/requirements.cpu.txt

RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r /app/requirements.cpu.txt

# Copier tout le dossier backend (contexte) vers /app/backend
COPY . /app/backend

EXPOSE 8000
CMD ["uvicorn", "backend.CounterArg_Service.app:app", "--host", "0.0.0.0", "--port", "8000"]
