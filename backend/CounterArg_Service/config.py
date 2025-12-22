import os
from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    # évite le warning "model_name conflict"
    model_config = ConfigDict(protected_namespaces=())

    # Modèle (LLM) -> oui, c'est bien le "nom du modèle" que tu vas charger
    model_name: str = os.getenv("MODEL_NAME", "google/flan-t5-base")

    # Génération
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))

    # Device
    device: str = os.getenv("DEVICE", os.getenv("device", "cpu"))
    use_gpu: bool = os.getenv("USE_GPU", "false").lower() == "true"

    # Compatibilité avec ton ancien code (settings.DEVICE)
    @property
    def DEVICE(self) -> str:
        return self.device


settings = Settings()
