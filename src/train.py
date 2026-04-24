import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import classification_report, roc_auc_score
import joblib

def load_and_preprocess(path: str):
    df = pd.read_csv(path)

    # remove customerId, pois não tem valor preditivo
    df = df.drop('customerID', axis = 1)

    # remove tenure = 0, que indicam clientessem histórico
    n_zeros = (df['tenure'] == 0).sum()
    print(f"Removidos {n_zeros} registros com tenure = 0")
    df = df[df['tenure'] > 0]

    # recodifica SeniorCitizen para ser categórica
    df['SeniorCitizen'] = df['SeniorCitizen'].map({
        0: 'No',
        1: 'Yes'
        })

    # remove TotalCharges por alta multicolinearidade com tenure
    df = df.drop('TotalCharges', axis = 1)

    # remove PhoneService por baixa correlação com Churn
    df = df.drop('PhoneService', axis = 1)

    # converte churn para 0 e 1
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # colunas numéricas normalizadas
    num_cols = ['tenure', 'MonthlyCharges']

    # colunas categórias em dummies
    cat_cols = df.select_dtypes(include = 'object').columns.tolist()
    df = pd.get_dummies(df, columns = cat_cols, drop_first = True)

    # split e normalização
    X = df.drop('Churn', axis = 1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42, stratify = y
    )

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    joblib.dump(scaler, 'models/scaler.pk1')

    print(f"Features finais: {X_train.shape[1]}")
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")
    print(f"Churn no treino: {y_train.mean():.1%}")
    print(f"Churn no teste: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test

def train_and_compare(X_train, X_test, y_train, y_test):

    modelos = {
        'Logistic Regression': LogisticRegression(max_iter = 1000, random_state = 42),
        'Random Forest': RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs = -1)
    }

    resultados = {}

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        resultados[nome] = (auc, modelo)

        print(f"\n{'=' * 50}")
        print(f"{nome}")
        print(f"\n{'=' * 50}")
        print(classification_report(y_test, y_pred, target_names = ['Ficou', 'Cancelou']))
        print(f"ROC-AUC: {auc:.4f}")

    melhor_nome = max(resultados, key = lambda k: resultados[k][0])
    melhor_modelo = resultados[melhor_nome][1]
    joblib.dump(melhor_modelo, 'models/best_model.pk1')

    print(f"\n{'=' * 50}")
    print(f"Melhor: {melhor_nome}")
    print(f"AUC = {resultados[melhor_nome][1]}")
    print("Modelo salvo em models/best_model.pk1")

    return melhor_modelo

if __name__ == "__main__":
    print("=" * 50)
    print("Carregando e pré-processando dados...")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_and_preprocess('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    print("\n Pré-processamento concluído!")

    print("=" * 50)
    print("Treinando modelos..")
    print("=" * 50)

    model = train_and_compare(X_train, X_test, y_train, y_test)

    print("\n Treinamento concluído!")