import mlflow

def main():
    mlflow.set_experiment("counter-argument-generation")

    with mlflow.start_run():
        # Paramètres
        mlflow.log_param("model_name", "google/flan-t5-base")
        mlflow.log_param("temperature", 0.7)
        mlflow.log_param("max_new_tokens", 256)

        # Métriques (simples pour commencer)
        mlflow.log_metric("latency_ms", 120)
        mlflow.log_metric("format_ok_rate", 1.0)

        # Artefact
        with open("prompt_v1.txt", "w") as f:
            f.write("CLAIM: example\nCOUNTER-ARGUMENT:")
        mlflow.log_artifact("prompt_v1.txt")

if __name__ == "__main__":
    main()
