FROM python:3.12-slim
WORKDIR /app

COPY requirements.cpu.txt .

# torch CPU-only (évite CUDA + nvidia-*)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

RUN pip install --no-cache-dir -r requirements.cpu.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.CounterArg_Service.app:app", "--host", "0.0.0.0", "--port", "8000"]
