# Embeddings Multimodales - Documentación

## 📚 Descripción

Este módulo proporciona embeddings multimodales de alta calidad para texto, imágenes y audio usando modelos open-source de última generación cargados localmente.

## 🎯 Modelos Soportados

| Modalidad | Modelo | Dimensiones | Librería |
|-----------|--------|-------------|----------|
| **Texto** | `BAAI/bge-m3` | 1024 | sentence-transformers |
| **Imagen** | `google/siglip-so400m-patch14-384` | 1152 | transformers |
| **Audio** | `laion/clap-htsat-unfused` | 512 | transformers |

## 🚀 Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. (Opcional) Instalar PyTorch con soporte CUDA

Para GPU NVIDIA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Descargar modelos (automático)

Los modelos se descargarán automáticamente la primera vez que inicies el servidor.

## 📡 Iniciar el Servidor

```bash
python main.py
```

El servidor se iniciará en `http://localhost:8765`

## 🔧 Uso de la API

### 1. Información de Modelos

```bash
curl http://localhost:8765/v1/embeddings/models
```

**Respuesta:**
```json
{
  "device": "cuda",
  "models": {
    "text": {
      "model_id": "BAAI/bge-m3",
      "dimensions": 1024,
      "library": "sentence-transformers"
    },
    "image": {
      "model_id": "google/siglip-so400m-patch14-384",
      "dimensions": 1152,
      "library": "transformers"
    },
    "audio": {
      "model_id": "laion/clap-htsat-unfused",
      "dimensions": 512,
      "library": "transformers",
      "sample_rate": 48000
    }
  }
}
```

---

### 2. Embedding de Texto

**Endpoint:** `POST /v1/embeddings/text`

```bash
curl -X POST http://localhost:8765/v1/embeddings/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "El perro corre por el parque",
    "normalize": true
  }'
```

**Respuesta:**
```json
{
  "object": "embedding",
  "model": "BAAI/bge-m3",
  "embedding": [0.023, -0.145, 0.789, ...], // 1024 valores
  "dimensions": 1024
}
```

---

### 3. Embedding de Texto en Batch

**Endpoint:** `POST /v1/embeddings/text/batch`

```bash
curl -X POST http://localhost:8765/v1/embeddings/text/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Primera frase",
      "Segunda frase",
      "Tercera frase"
    ],
    "normalize": true
  }'
```

**Respuesta:**
```json
{
  "object": "list",
  "model": "BAAI/bge-m3",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [...]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [...]
    }
  ],
  "total": 3
}
```

---

### 4. Embedding de Imagen

**Endpoint:** `POST /v1/embeddings/image`

```bash
curl -X POST http://localhost:8765/v1/embeddings/image \
  -F "file=@imagen.jpg" \
  -F "normalize=true"
```

**Respuesta:**
```json
{
  "object": "embedding",
  "model": "google/siglip-so400m-patch14-384",
  "embedding": [...], // 1152 valores
  "dimensions": 1152
}
```

**Formatos soportados:** JPG, PNG, WEBP, BMP, etc.

---

### 5. Embedding de Audio

**Endpoint:** `POST /v1/embeddings/audio`

```bash
curl -X POST http://localhost:8765/v1/embeddings/audio \
  -F "file=@audio.wav" \
  -F "normalize=true" \
  -F "max_duration=10.0"
```

**Respuesta:**
```json
{
  "object": "embedding",
  "model": "laion/clap-htsat-unfused",
  "embedding": [...], // 512 valores
  "dimensions": 512
}
```

**Formatos soportados:** WAV, MP3, FLAC, OGG, M4A, etc.

**Parámetros opcionales:**
- `max_duration`: Limita la duración del audio procesado (en segundos)

---

## 🧪 Pruebas

Ejecuta el script de pruebas:

```bash
python test_embeddings.py
```

Para probar con archivos reales, modifica el script:

```python
# En test_embeddings.py
test_image_embedding('ruta/a/tu/imagen.jpg')
test_audio_embedding('ruta/a/tu/audio.wav')
```

---

## 🐍 Uso desde Python

### Ejemplo 1: Texto

```python
import requests

response = requests.post(
    "http://localhost:8765/v1/embeddings/text",
    json={
        "text": "Buscar información sobre inteligencia artificial",
        "normalize": True
    }
)

embedding = response.json()["embedding"]
print(f"Vector de {len(embedding)} dimensiones")
```

### Ejemplo 2: Imagen

```python
import requests

with open("imagen.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8765/v1/embeddings/image",
        files={"file": f}
    )

embedding = response.json()["embedding"]
print(f"Vector de imagen: {len(embedding)} dimensiones")
```

### Ejemplo 3: Similitud Coseno

```python
import numpy as np

def cosine_similarity(v1, v2):
    """Calcula similitud coseno entre dos vectores normalizados"""
    return np.dot(v1, v2)

# Generar embeddings
text1 = get_embedding("El perro corre")
text2 = get_embedding("Un canino corriendo")
text3 = get_embedding("La luna brilla")

# Calcular similitudes
sim_1_2 = cosine_similarity(text1, text2)  # Alta similitud (~0.8)
sim_1_3 = cosine_similarity(text1, text3)  # Baja similitud (~0.2)

print(f"Similitud perro/canino: {sim_1_2:.3f}")
print(f"Similitud perro/luna: {sim_1_3:.3f}")
```

---

## 📊 Características Técnicas

### BGE-M3 (Texto)
- ✅ Multilingüe (soporta más de 100 idiomas)
- ✅ Optimizado para búsqueda semántica
- ✅ State-of-the-art en benchmarks RAG
- ✅ Soporta textos largos (hasta 8192 tokens)

### SigLIP (Imagen)
- ✅ Modelo vision-language de Google
- ✅ Entrenado con pares imagen-texto
- ✅ Excelente para búsqueda multimodal
- ✅ Maneja imágenes de 384x384

### CLAP (Audio)
- ✅ Modelo audio-language
- ✅ Captura características semánticas del audio
- ✅ Funciona con música, voz y sonidos ambientales
- ✅ Resampleo automático a 48kHz

---

## ⚙️ Configuración Avanzada

### Modificar modelos en `config.py`:

```python
EMBEDDING_MODELS = {
    "text": {
        "model_id": "BAAI/bge-m3",  # Cambia por otro modelo
        "library": "sentence-transformers",
        "dimensions": 1024
    },
    # ...
}
```

### Parámetros de normalización

Por defecto, todos los vectores están normalizados (L2 norm = 1). Esto permite:
- ✅ Usar producto punto en lugar de cosine similarity (más rápido)
- ✅ Comparaciones directas entre vectores
- ✅ Integración con bases de datos vectoriales (Weaviate, Qdrant, etc.)

---

## 🔍 Documentación Interactiva

Accede a la documentación Swagger en:
```
http://localhost:8765/docs
```

---

## 📦 Integración con Bases de Datos Vectoriales

### Weaviate

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Insertar con embedding
client.data_object.create(
    {
        "text": "Mi documento",
        "vector": embedding  # Vector de 1024 dim
    },
    "Document"
)

# Búsqueda por similitud
result = client.query.get("Document", ["text"]).with_near_vector({
    "vector": query_embedding
}).with_limit(5).do()
```

### Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient("localhost", port=6333)

# Crear colección
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

# Insertar
client.upsert(
    collection_name="documents",
    points=[{
        "id": 1,
        "vector": embedding,
        "payload": {"text": "Mi documento"}
    }]
)
```

---

## 🎯 Casos de Uso

1. **Búsqueda Semántica**: Encuentra documentos similares por significado
2. **RAG (Retrieval Augmented Generation)**: Alimenta LLMs con contexto relevante
3. **Clasificación Zero-Shot**: Clasifica sin necesidad de entrenamiento
4. **Clustering**: Agrupa documentos/imágenes/audios similares
5. **Deduplicación**: Detecta contenido duplicado o similar
6. **Búsqueda Multimodal**: Busca imágenes con texto, o viceversa

---

## 🐛 Troubleshooting

### Error: CUDA out of memory

**Solución:** Usa CPU en lugar de GPU
```python
# En services/embedding_service.py
_device: str = "cpu"  # Forzar CPU
```

### Error: Modelo no descarga

**Solución:** Descarga manual
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

### Audio con errores

**Solución:** Verifica el formato
```bash
ffmpeg -i audio_original.mp3 -ar 48000 -ac 1 audio_convertido.wav
```

---

## 📄 Licencia

Los modelos tienen licencias open-source:
- BGE-M3: MIT License
- SigLIP: Apache 2.0
- CLAP: Apache 2.0
