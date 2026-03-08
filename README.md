# LLM Gateway API

API Gateway inteligente con soporte multimodal que enruta automáticamente entre modelos locales (Ollama) y cloud (Google Gemini) basándose en privacidad y tipo de tarea.

## 🎯 Características

- **🎭 Multimodal**: Chat, Vision, OCR con soporte para texto, imágenes, audio y video
- **📁 Archivos Directos**: Sube archivos multimedia directamente (hasta 100MB)
- **🔐 Privacy-First**: Modelos locales para datos sensibles, cloud para máximo rendimiento
- **🚀 Google File API**: Manejo inteligente de archivos grandes (>= 5MB)
- **📊 Embeddings**: API separada para embeddings de texto, imagen y audio
- **🔄 Compatible OpenAI**: Formato de API estándar
- **📚 FastAPI**: Documentación Swagger automática

---

## 📋 Requisitos

### Software
- **Python 3.10+**
- **Ollama** (para modelos locales)
- **NVIDIA GPU** (opcional, para embeddings acelerados)

### API Keys
- **Google Gemini API Key** (para `privacy_mode=flexible`)
- **OpenRouter API Key** (para modelos a través de OpenRouter)

### Modelos Ollama
```bash
ollama pull CognitiveComputations/dolphin-mistral-nemo:latest
ollama pull qwen3-vl:8b
ollama pull deepseek-ocr:3b
```

---

## 🚀 Instalación Rápida

```bash
# 1. Clonar e instalar
cd llm-endpoints
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Configurar API Key
copy .env.example .env
# Editar .env y agregar tu GEMINI_API_KEY

# 3. Iniciar servidor
python main.py
```

**Servidor:** http://localhost:8765  
**Docs:** http://localhost:8765/docs

---

## 📚 Endpoints Principales

### 1. Chat Completions `/v1/chat/completions`

Endpoint multimodal con soporte para archivos adjuntos.

**Formato:** `multipart/form-data`

**Parámetros:**
- `task`: `"chat"` | `"vision"` | `"ocr"`
- `privacy_mode`: `"strict"` (local) | `"flexible"` (cloud)
- `messages`: JSON string con mensajes
- `files`: Archivos multimedia (opcional)

**Límites de archivos:**
| Tamaño | Estrategia | Privacy Mode |
|--------|------------|--------------|
| < 5MB | Base64 | Cualquiera |
| >= 5MB | Google File API | `flexible` |

#### Ejemplo 1: Chat Simple

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=chat' \
  -F 'privacy_mode=strict' \
  -F 'messages=[{"role":"user","content":"Resume este texto confidencial"}]'
```

#### Ejemplo 2: Vision con Imagen

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=vision' \
  -F 'privacy_mode=flexible' \
  -F 'messages=[{"role":"user","content":[{"type":"text","text":"¿Qué ves?"},{"type":"image","file_index":0}]}]' \
  -F 'files=@imagen.jpg'
```

#### Ejemplo 3: Audio Grande (Google File API)

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=vision' \
  -F 'privacy_mode=flexible' \
  -F 'messages=[{"role":"user","content":[{"type":"text","text":"Transcribe"},{"type":"audio","file_index":0}]}]' \
  -F 'files=@audio_large.mp3'
```

---

### 2. Embeddings API

Endpoints separados para embeddings de texto, imagen y audio.

#### 2.1 Texto `/v1/embeddings/text`

```bash
curl -X POST http://localhost:8765/v1/embeddings/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Un texto de ejemplo", "normalize": true}'
```

**Modelo:** `BAAI/bge-m3` (1024 dimensiones)

#### 2.2 Imagen `/v1/embeddings/image`

```bash
curl -X POST http://localhost:8765/v1/embeddings/image \
  -F "file=@imagen.jpg"
```

**Modelo:** `google/siglip-so400m-patch14-384` (1152 dim)

#### 2.3 Audio `/v1/embeddings/audio`

```bash
curl -X POST http://localhost:8765/v1/embeddings/audio \
  -F "file=@audio.wav"
```

**Modelo:** `laion/clap-htsat-unfused` (512 dim)

**Ver:** [`EMBEDDINGS.md`](docs/EMBEDDINGS.md) para documentación completa

---

## 🗺️ Routing de Modelos

### Chat Completions

| Task | Privacy: Strict | Privacy: Flexible |
|------|----------------|-------------------|
| **chat** | `ollama/dolphin-mistral-nemo` | `gemini/gemini-2.5-flash` |
| **vision** | `ollama/qwen3-vl:8b` | `gemini/gemini-2.5-flash` |
| **ocr** | `ollama/deepseek-ocr:3b` | `gemini/gemini-2.5-flash` |

### Embeddings

| Modalidad | Modelo | Dimensiones |
|-----------|--------|-------------|
| **texto** | `BAAI/bge-m3` | 1024 |
| **imagen** | `google/siglip-so400m-patch14-384` | 1152 |
| **audio** | `laion/clap-htsat-unfused` | 512 |

---

## 📁 Estructura del Proyecto

```
llm-endpoints/
├── main.py                      # FastAPI app principal
├── config.py                    # Configuración y routing
├── requirements.txt             # Dependencias
├── .env.example                # Template de variables
│
├── routers/
│   ├── chat.py                 # Chat completions multimodal
│   └── embeddings.py           # Endpoints de embeddings
│
├── services/
│   ├── llm_client.py           # Cliente LiteLLM
│   ├── router.py               # Lógica de routing
│   ├── file_processor.py       # Procesamiento de archivos
│   ├── google_file_api.py      # Google File API client
│   └── embedding_service.py    # Servicio de embeddings
│
├── schemas/
│   ├── requests.py             # Schemas de request
│   └── responses.py            # Schemas de response
│
└── docs/
    ├── MULTIMODAL_CHAT.md      # Guía de uso multimodal
    ├── EMBEDDINGS.md           # Guía de embeddings
    ├── GEMINI_CONFIG.md        # Configuración de Gemini
    └── RTX4090_OPTIMIZATIONS.md # Optimizaciones GPU
```

---

## 🔧 Uso Avanzado

### Python SDK

```python
import requests
import json

# Chat simple
response = requests.post(
    "http://localhost:8765/v1/chat/completions",
    data={
        "task": "chat",
        "privacy_mode": "strict",
        "messages": json.dumps([
            {"role": "user", "content": "Hola"}
        ])
    }
)

print(response.json()["choices"][0]["message"]["content"])

# Vision con imagen
with open("imagen.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8765/v1/chat/completions",
        data={
            "task": "vision",
            "privacy_mode": "flexible",
            "messages": json.dumps([{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image", "file_index": 0}
                ]
            }])
        },
        files=[("files", ("imagen.jpg", f, "image/jpeg"))]
    )
```

---

## 🎓 Casos de Uso

### 1. RAG con Embeddings Multimodales
```python
# Generar embeddings de documentos con imágenes
text_emb = embed_text("Contenido del documento")
image_emb = embed_image("diagrama.png")

# Buscar en base de datos vectorial
results = vector_db.search(query_embedding, top_k=5)
```

### 2. Análisis de Imágenes Privado
```python
# OCR local de documentos sensibles
response = chat(
    task="ocr",
    privacy_mode="strict",  # ¡Sin enviar a cloud!
    messages=[...],
    files=["factura.png"]
)
```

### 3. Transcripción de Audio Largo
```python
# Audio de 10MB se sube automáticamente a Google File API
response = chat(
    task="vision",
    privacy_mode="flexible",
    messages=[...],
    files=["reunion_1hora.mp3"]  # Sube a Google, retorna URI
)
```

---

## 📊 Monitoreo y Health Check

```bash
# Health check
curl http://localhost:8765/health

# Listar modelos
curl http://localhost:8765/v1/models

# Info de embeddings
curl http://localhost:8765/v1/embeddings/models
```

---

## 🐛 Troubleshooting

### Error: "Modelo no encontrado"
```bash
# Verificar Ollama
ollama list

# Descargar modelo específico
ollama pull CognitiveComputations/dolphin-mistral-nemo:latest
```

### Error: Autenticación Gemini
```bash
# Verificar .env
cat .env | grep GEMINI_API_KEY

# Obtener nueva API key
# https://makersuite.google.com/app/apikey
```

### Archivo grande + privacy_mode=strict
```
Error 501: "Procesamiento local de archivos grandes en desarrollo"

Solución: Usar privacy_mode=flexible para archivos >= 5MB
```

---

## 📖 Documentación Adicional

- **[MULTIMODAL_CHAT.md](docs/MULTIMODAL_CHAT.md)** - Guía completa de chat multimodal
- **[EMBEDDINGS.md](docs/EMBEDDINGS.md)** - API de embeddings multimodales
- **[GEMINI_CONFIG.md](docs/GEMINI_CONFIG.md)** - Configurar Google Gemini
- **[RTX4090_OPTIMIZATIONS.md](docs/RTX4090_OPTIMIZATIONS.md)** - Optimizaciones GPU

---

## 🚀 Optimizaciones

### RTX 4090
- ✅ FP16 automático para embeddings
- ✅ cuDNN benchmark habilitado
- ✅ TF32 para matrix multiplications
- 📈 2-3x más rápido en inferencia

**Ver:** [`RTX4090_OPTIMIZATIONS.md`](docs/RTX4090_OPTIMIZATIONS.md)

---

## 🧪 Testing

```bash
# Tests de chat multimodal
python test_multimodal_chat.py

# Tests de embeddings
python test_embeddings.py
```

---

## 📝 Licencia

MIT

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Abre un issue o pull request.

---

## 🔗 Enlaces Útiles

- [Google AI Studio](https://makersuite.google.com/)
- [Ollama](https://ollama.ai/)
- [LiteLLM Docs](https://docs.litellm.ai/)
- [FastAPI](https://fastapi.tiangolo.com/)
