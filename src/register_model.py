import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "churn-classifier" # Nome do modelo para registro
client = MlflowClient() # Criando um cliente Mlflow (API)

"""Busca a run com a maior métrica no experimento"""
def get_best_run(experiment_name: str, metric: str = "roc_auc"):
    # registrando o experimento pelo nome 
    exp = client.get_experiment_by_name(experiment_name)

    # registrando as runs do experimento, ordenando pelas métricas
    runs = client.search_runs(
        experiment_ids = [ exp.experiment_id ],
        order_by = [ f"metrics.{ metric } DESC" ],
        max_results = 1
    )

    # salvando e retornando a melhor run
    best = runs[0]
    print(f"Melhor run: { best.info.run_name }")
    print(f"`{ metric } = {best.data.metrics[metric]:.4f}")
    print(f"run_id = { best.info.run_id }")
    
    return best

"""Registra o modelo no MLflow Model Registry"""
def register_model(run_id: str):
    # definindo uri do modelo
    model_uri = f"runs:/{run_id}/model"

    # registrando e retornando modelo
    result = mlflow.register_model(
        model_uri = model_uri,
        name = MODEL_NAME
    )

    print(f"\n Modelo registrado: { MODEL_NAME } v{ result.version }")
    return result.version

"""Promove a versão para Staging."""
def promote_to_staging(version: str): 
    # promove o modelo p
    client.transition_model_version_stage(
        name = MODEL_NAME,
        version = version,
        stage = "Staging"
    )

    print(f"v{ version } promovido para Staging")
    '''
    PARA NÃO ESQUECER: Staging é um estágio predefinido no
    Model Registry usado para validar e testar uma versão específica
    de um modelo de machine learning. É basicamente o modelo saindo 
    da fase de desenvolvimento para uma fase de validação!
    '''
    # Ciclo de vida do MLflow Registry: None --> Staging --> Produciton --> Archived

if __name__ == "__main__":
    best = get_best_run("churn-prediction")
    version = register_model(best.info.run_id)
    promote_to_staging(version)

    print(f"\n Veja no UI: http://127.0.0.1:5000/#/models/{MODEL_NAME}")