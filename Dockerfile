# Usamos python 3.10 slim como base
FROM python:3.10-slim

# Variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Evitar problemas con directorios de cache en build
    PIP_NO_CACHE_DIR=off

# Instalar dependencias del sistema necesarias
# libsndfile1: para soundfile
# ffmpeg: para procesamiento de audio/video
# build-essential: para compilar algunas librerías de python si es necesario
# git: por si se instalan dps desde git
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar cache de Docker layers
COPY requirements.txt .

# Instalar dependencias de Python
# Actualizamos pip primero
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto
EXPOSE 8765

# Comando por defecto (será sobreescrito por docker-compose para dev)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]
