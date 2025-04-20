FROM python:3.9-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia o requirements.txt e instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia toda a estrutura do projeto (incluindo a pasta src)
COPY . .

# Define o comando de entrada apontando para src/main.py
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "60"]