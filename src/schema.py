from pydantic import BaseModel, Field
from typing import Literal

"""
    Dados de entrada para predição de churn.
    Todos os campos espelham as colunas do dataset original
    após remoção de customerID, TotalCharges e PhoneService.
"""
class CustomerInput(BaseModel):

    # dados numéricos
    tenure: int = Field(..., ge = 1, le = 72, description = 'Meses de contrato')
    MonthlyCharges: float = Field(..., ge = 0, le = 200, description = 'Valor mensal em USD')

    # valores categóricos (Literal garante que somente valores válidos serão aceitos)
    SeniorCitizen: Literal['No', 'Yes']
    Partner: Literal['No', 'Yes']
    Dependents: Literal['No', 'Yes']
    MultipleLines: Literal['No', 'Yes', 'No phone service']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: Literal['No', 'Yes', 'No internet service']
    OnlineBackup: Literal['No', 'Yes', 'No internet service']
    DeviceProtection: Literal['No', 'Yes', 'No internet service']
    TechSupport: Literal['No', 'Yes', 'No internet service']
    StreamingTV: Literal['No', 'Yes', 'No internet service']
    StreamingMovies: Literal['No', 'Yes', 'No internet service']
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling: Literal['No', 'Yes']
    PaymentMethod: Literal[
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ]

"""
    Exemplo da documentação automática do FastAPI.
    Foram definidos valores mais frequentes (moda de cada variável)
    ou valores que influeciam de forma neutra o churn
"""
class Config:
    json_schema_extra = {
        "example": {
            "tenure": 6,
            "MonthlyCharges": 70.5,
            "SeniorCitizen": "No",
            "Partner": "No",
            "Dependents": "No",
            "MultipleLine": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Eletronic check"
        }
    }

"""
    Resposta da API
"""
class PredictionOutput(BaseModel):
    churn_probability: float = Field(description = "Probabilidade de churn (0 a 1)")
    churn_prediction: bool = Field(description = "True se o cliente tem aalto risco de churn")
    risk_level: str = Field(description = "low | medium | high")
