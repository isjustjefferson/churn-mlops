import requests

URL_API = "https://churn-mlops-production.up.railway.app"

payload = {
    "tenure": 6,
    "MonthlyCharges": 70.5,
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}

r = requests.post(f"{URL_API}/predict", json=payload)
print("Status:", r.status_code)
print("Resposta:", r.json())