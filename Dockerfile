# imagem base
FROM python:3.14.3-slim

# define o diretório de trabalhp
WORKDIR /app

# copia o requirements.txt
COPY requirements.txt .

# instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# copia arquivos dos diretórios src e models
COPY src/ ./src/
COPY models/ ./models/

# expõe porta da API
EXPOSE 8000

# comando executado ao iniciar o container
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]