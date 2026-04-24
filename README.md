# 📊 Churn MLOps Pipeline: Da EDA ao Deployment

[![Python](https://img.shields.io/badge/Python-3+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.org/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

Este repositório contém a implementação de um pipeline de **MLOps** ponta a ponta para a predição de rotatividade de clientes (*Churn*). O projeto demonstra a transição de um modelo experimental em um Jupyter Notebook para uma aplicação conteinerizada e pronta para produção.

## 🎯 Visão do Projeto
O Churn é uma das métricas mais críticas para empresas de serviços. Este projeto utiliza o dataset *Telco Customer Churn* para identificar padrões de comportamento e prever quais clientes têm maior probabilidade de cancelar seus serviços, permitindo ações preventivas de retenção.

---

## 🛤️ Roadmap de Desenvolvimento

- [x] **Fase 1: Baseline & EDA** - Definição do problema, limpeza de dados e treinamento de modelos iniciais.
- [x] **Fase 2: MLOps com MLflow** - Implementação de rastreamento de experimentos, versionamento de parâmetros e métricas.
- [ ] **Fase 3: Model Serving** - Criação de uma API REST robusta utilizando **FastAPI**.
- [ ] **Fase 4: Containerização** - Empacotamento da aplicação com **Docker** para garantir portabilidade.
- [ ] **Fase 5: Cloud Deployment** - Deploy automatizado via CI/CD em plataformas Cloud (Railway/Render).

---

## 🛠️ Stack Tecnológica
* **Linguagem:** Python 3+
* **Data Science:** Pandas, Numpy, Scikit-Learn.
* **Visualização:** Matplotlib, Seaborn.
* **MLOps:** MLflow (Tracking & Registry).
* **Backend:** FastAPI (Uvicorn).
* **DevOps:** Docker, Git.

---

## 📈 Insights de Engenharia & Dados

Durante a análise exploratória, foram aplicadas técnicas fundamentais para garantir a saúde do modelo:
* **Tratamento de Multicolinearidade:** Remoção da variável `TotalCharges` devido à alta correlação (0.83) com `tenure`, evitando viés em modelos lineares.
* **Feature Engineering:** Uso de **One-Hot Encoding** para transformar variáveis categóricas em representações numéricas interpretáveis pelo modelo.
* **Análise de Correlação:** Identificou-se que contratos do tipo `Month-to-month` são os maiores preditores de cancelamento.

###  ⚖ Comparativo de Performance (MLflow)

| Modelo | Accuracy | ROC AUC | Recall (Churn) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 0.80 | 0.83 | 0.51 |
| **Random Forest (Baseline)** | 0.78 | 0.82 | 0.57 |
| **Random Forest (Tuned)** | 0.80 | 0.83 | 0.49 |

---

## 🏗️ Estrutura do Repositório

```bash
├── data/                     # Datasets (Raw e Processed)
├── mlruns/                   # Logs e artefatos de experimentos
├── models/                   # best_model e scaler (Pipeline antiga de treinamento)
├── notebooks/                # EDA e Prototipagem de modelos
├── src/                      # Scripts de produção (Treino e API)
│   ├── train.py              # Pipeline de treinamento e log no MLflow
│   └── register_model.py     # Pipeline de registro do melhor modelo no MLflow
├── README.md                 # Documentação do projeto
├── mlflow.db                 # Banco de dados local MLflow
└── requirements.txt          # Gerenciamento de dependências
```
---

## 🚀 Como Executar o Projeto
### ⚙️ Requisitos:
- **Python 3+**
- **Git** (ou GitHub Desktop)
### 📝 Passo a passo:
1. Clone o repositório:
```bash
git clone https://github.com/isjustjefferson/churn-mlops.git
cd churn-mlops
```
> Caso esteja usando o GitHub Desktop, siga o passo a passo da aplicação para clonar o repositório: https://docs.github.com/pt/desktop/adding-and-cloning-repositories/cloning-a-repository-from-github-to-github-desktop
2. Criar um ambiente virtual (Recomendado, não é obrigatório para testar o projeto):
```bash
python3 -m venv .venv      # Windows: python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
3. Instale as dependências:
```bash
pip3 install -r requirements.txt  # Windows: pip install -r requirements.txt
```
4. Treine os modelos:
```bash
python3 src/train.py # Windows: python src/train.py
```
> Caso tenha criado um **ambiente virtual (venv)**, sempre é necessário iniciá-lo pelo comando ```source .venv/bin/activate``` (Linux) ou ```.venv/Scripts/activate``` (Windows).
5. Visualize no dashboad do MLflow UI:
```bash
mlflow ui
```
> Clique no experimento ``churn-prediction`` e explore. Recomenda-se usar ``Compare`` entre os modelos, visualizar parâmetros, métricas e o modelo em ``Artifacts`` e observar os gráficos comparativo entre as métricas dos modelos na aba ``Chart view``
6. (Recomendado) Registre o melhor modelo no Model Registry:
```bash
python3 src/register_model.py  # Windows: python src/register_model.py
```
> **Atenção ao terminal!** Ele irá informar o URL do Model Registry no MLflow UI:
> ```bash
> Veja no UI: http://127.0.0.1:5000/#/models/churn-classifier
> ```
