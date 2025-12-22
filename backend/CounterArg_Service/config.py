from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    # évite le warning "model_name conflict with protected namespace model_"
    model_config = ConfigDict(protected_namespaces=())

    # LLM / modèle HF
    model_name: str = "google/flan-t5-base"

    # génération
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # exécution
    use_gpu: bool = False
    device: str = "cpu"  # "cpu" ou "cuda"

    @property
    def DEVICE(self) -> str:
        """
        Compatibilité avec ton ancien code (settings.DEVICE).
        """
        return self.device


settings = Settings()


def load_best_config() -> Settings:
    """
    Placeholder: si plus tard tu veux charger une config depuis MLflow / fichier.
    Pour l’instant, on retourne la config par défaut.
    """
    return settings
