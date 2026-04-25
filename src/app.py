import joblib
import pandas as pd 
import numpy as np 
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.schema import CustomerInput, PredictionOutput

"""
    CARREGAMENTO DE MODELO
"""
model = None
scaler = None

# colunas na mesma ordem que o modelo foi treinado
FEATURE_COLUMNS = None

"""Carrega modelo e scaler salvos"""
def load_artifacts():
    global model, scaler
    model = joblib.load('models/best_model.pk1')
    scaler = joblib.load('models/scaler.pk1')
    print("Modelo carregado: ", type(model).__name__)
    print("Scaler carregado")

"""
    Aplica o mesmo pré-processamento do train.py no input da API.
    Garante que as features chegam ao modelo no formato correto.
"""
def preprocess_input(data: CustomerInput) -> pd.DataFrame:
    # convertendo o input Pydantic para DataFrame
    df = pd.dataFrame([data.dict()])

    # carrregando colunas numéricas
    num_cols = ['tenure', 'MonthlyCharges']

    # carregando colunas categóricas
    cat_cols = df.select_dtypes(include = 'object').columns.tolist()
    df = pd.get_dummies(df, columns = cat_cols, drop_first = True)

    # carregando colunas de treino
    train_cols = model.feature_names_in_
    df = df.reindez(columns = trains_cols, fill_value = 0) # colunas que não existem recebem input 0

    # normalizando os valores numéricos
    df[num_cols] = scaler.transform(df[num_cols])

    return df

"""Classifica o risco baseado na probabilidade."""
def get_risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    elif probability >= 0.4:
        return "medium"
    else:
        return "low"

"""
    CARREGAMENTO DA API
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts() # Roda ao iniciar
    yield # Roda ao encerrar

app = FastAPI(
    title = "Churn Prediction API",
    description = "Prediz a probabilidade de churn de clientes de telecom.",
    version = "1.0.0",
    lifespan = lifespan
)

# ENDPOINTS
@app.get("/")

def health_check():
    """
    Verifica se a API está rodando e o modelo foi carregado.
    """
    return {
        "status": "online",
        "model": type(model).__name__ if model else "not loaded",
        "version": "1.0.0"
    }

@app.post("/predict", response_model = PredictionOutput)

def predict(customer: CustomerInput):
    """
    Recebe os dados de um cliente e retorna:
    - churn_probability: probabilidade entre 0 e 1
    - churn_prediction: True se probabilidade >= 0.5
    - risk_level: low | medium | high
    """
    # lançando erro HTTP 503 (Service Unavailable)
    if model is None:
        raise HTTPException(status_code = 503, detail = "Modelo não carregado")

    try: 
        # carregando dados recebido pelo cliente para predição
        df = preprocess_input(customer)
        probability = float(model.predict_proba(df)[0][1])
        prediction = probability >= 0.5
        risk = get_risk_level(probability)

        # retornando predição
        return PredictionOutput(
            churn_probability = round(probability, 4),
            churn_prediction = prediction,
            risk_level = risk
        )
    except Exception as e:
        # lançando erro 500 (Internal Server Error) <- resposta genérica, melhorar posteriormente
        raise HTTPException(status_code = 500, detail = f"Erro na predição: {str(e)}")