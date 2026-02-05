# API Documentation

Documentación completa de endpoints disponibles en LLM Gateway API.

---

## Base URL

```
http://localhost:8765
```

---

## 📚 Tabla de Contenido

1. [Chat Completions](#1-chat-completions)
2. [Embeddings API](#2-embeddings-api)
3. [Utility Endpoints](#3-utility-endpoints)
4. [Error Codes](#4-error-codes)

---

## 1. Chat Completions

### `POST /v1/chat/completions`

Endpoint multimodal para chat, análisis de imágenes y OCR.

**Formato:** `multipart/form-data`

#### Request Parameters

| Campo | Tipo | Required | Descripción |
|-------|------|----------|-------------|
| `task` | string | ✅ | `"chat"`, `"vision"`, `"ocr"` |
| `privacy_mode` | string | ✅ | `"strict"` (local) o `"flexible"` (cloud) |
| `messages` | JSON string | ✅ | Array de mensajes |
| `files` | File[] | ❌ | Archivos multimedia |
| `temperature` | float | ❌ | 0.0-2.0 (default: 0.7) |
| `max_tokens` | integer | ❌ | Máximo tokens a generar |
| `model` | string | ❌ | Override de modelo |

#### Messages Format

**Texto simple:**
```json
[
  {"role": "user", "content": "Hola"}
]
```

**Multimodal:**
```json
[
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "¿Qué ves?"},
      {"type": "image", "file_index": 0}
    ]
  }
]
```

#### File Handling

| Tamaño | Método | Limits |
|--------|--------|--------|
| < 5MB | Base64 data URI | Max: 5MB |
| >= 5MB | Google File API | Max: 100MB |

**Nota:** Archivos >= 5MB requieren `privacy_mode=flexible`

#### Response

```json
{
  "id": "chatcmpl-xyz",
  "object": "chat.completion",
  "created": 1704398400,
  "model": "gemini/gemini-2.5-flash",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "La respuesta..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  }
}
```

#### Examples

**Chat Simple:**
```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=chat' \
  -F 'privacy_mode=strict' \
  -F 'messages=[{"role":"user","content":"Hola"}]'
```

**Vision con Imagen:**
```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=vision' \
  -F 'privacy_mode=flexible' \
  -F 'messages=[{"role":"user","content":[{"type":"text","text":"Describe"},{"type":"image","file_index":0}]}]' \
  -F 'files=@imagen.jpg'
```

**OCR:**
```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=ocr' \
  -F 'privacy_mode=strict' \
  -F 'messages=[{"role":"user","content":[{"type":"text","text":"Extrae texto"},{"type":"image","file_index":0}]}]' \
  -F 'files=@documento.png'
```

---

## 2. Embeddings API

### 2.1 `POST /v1/embeddings/text`

Genera embedding de texto.

**Request:**
```json
{
  "text": "Un texto de ejemplo",
  "normalize": true
}
```

**Response:**
```json
{
  "object": "embedding",
  "model": "BAAI/bge-m3",
  "embedding": [0.023, -0.145, ...],
  "dimensions": 1024
}
```

#### Example

```bash
curl -X POST http://localhost:8765/v1/embeddings/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Ejemplo", "normalize": true}'
```

---

### 2.2 `POST /v1/embeddings/text/batch`

Genera embeddings de múltiples textos.

**Request:**
```json
{
  "texts": ["Texto 1", "Texto 2", "Texto 3"],
  "normalize": true
}
```

**Response:**
```json
{
  "object": "list",
  "model": "BAAI/bge-m3",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [...]},
    {"object": "embedding", "index": 1, "embedding": [...]},
    {"object": "embedding", "index": 2, "embedding": [...]}
  ],
  "total": 3
}
```

---

### 2.3 `POST /v1/embeddings/image`

Genera embedding de imagen.

**Request:** `multipart/form-data`
```
file: imagen.jpg
normalize: true
```

**Response:**
```json
{
  "object": "embedding",
  "model": "google/siglip-so400m-patch14-384",
  "embedding": [...],
  "dimensions": 1152
}
```

#### Example

```bash
curl -X POST http://localhost:8765/v1/embeddings/image \
  -F "file=@imagen.jpg" \
  -F "normalize=true"
```

---

### 2.4 `POST /v1/embeddings/audio`

Genera embedding de audio.

**Request:** `multipart/form-data`
```
file: audio.wav
normalize: true
max_duration: 10.0
```

**Response:**
```json
{
  "object": "embedding",
  "model": "laion/clap-htsat-unfused",
  "embedding": [...],
  "dimensions": 512
}
```

#### Example

```bash
curl -X POST http://localhost:8765/v1/embeddings/audio \
  -F "file=@audio.wav" \
  -F "normalize=true"
```

---

### 2.5 `GET /v1/embeddings/models`

Obtiene información de modelos de embeddings.

**Response:**
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

## 3. Utility Endpoints

### 3.1 `GET /health`

Health check del servidor.

**Response:**
```json
{"status": "ok"}
```

---

### 3.2 `GET /v1/models`

Lista modelos configurados para chat completions.

**Response:**
```json
{
  "available_tasks": ["chat", "vision", "ocr"],
  "model_configuration": {
    "chat": {
      "strict": "ollama/dolphin-mistral-nemo:latest",
      "flexible": "gemini/gemini-2.5-flash"
    },
    "vision": {
      "strict": "ollama/qwen3-vl:8b",
      "flexible": "gemini/gemini-2.5-flash"
    },
    "ocr": {
      "strict": "ollama/deepseek-ocr:3b",
      "flexible": "gemini/gemini-2.5-flash"
    }
  }
}
```

---

## 4. Error Codes

### HTTP Status Codes

| Code | Name | Descripción |
|------|------|-------------|
| 200 | OK | Éxito |
| 400 | Bad Request | Parámetros inválidos |
| 401 | Unauthorized | API key inválida |
| 404 | Not Found | Endpoint no existe |
| 429 | Too Many Requests | Rate limit excedido |
| 500 | Internal Server Error | Error del servidor |
| 501 | Not Implemented | Funcionalidad no implementada |

### Error Response Format

```json
{
  "detail": "Descripción del error"
}
```

### Common Errors

#### 1. Task Inválido
```json
{
  "detail": "Task inválido: embedding. Debe ser: chat, vision, ocr"
}
```

#### 2. Archivo Grande + Strict Mode
```json
{
  "detail": "Procesamiento local de archivos grandes (7.5MB) está en desarrollo.\n\nTODO Roadmap:..."
}
```

#### 3. File Index Inválido
```json
{
  "detail": "file_index 99 inválido. Se esperaban 1 archivos."
}
```

#### 4. Autenticación Gemini
```json
{
  "detail": "Error de autenticación. Verifica tu GEMINI_API_KEY en .env"
}
```

#### 5. Archivo Demasiado Grande
```json
{
  "detail": "Archivo demasiado grande: 150.0MB. Máximo permitido: 100.0MB"
}
```

---

## 5. Rate Limits

### Ollama (Local)
- Sin límites (depende de tu hardware)

### Google Gemini
- Según tu plan de Google AI
- Free tier: ~60 requests/minute
- Ver: https://ai.google.dev/pricing

---

## 6. Best Practices

### 1. Privacy Mode Selection

**Usar `strict` cuando:**
- Datos sensibles o confidenciales
- Cumplimiento de GDPR/privacidad
- No requieres modelos más avanzados

**Usar `flexible` cuando:**
- Máximo rendimiento requerido
- Archivos grandes (>= 5MB)
- Multimodal avanzado (video, audio largo)

### 2. File Uploads

**Optimizar tamaño:**
```bash
# Comprimir imagen
ffmpeg -i input.jpg -quality 85 output.jpg

# Comprimir audio
ffmpeg -i input.wav -b:a 128k output.mp3
```

**Límites recomendados:**
- Imágenes: < 2MB (usar JPEG con calidad 80-90)
- Audio: < 10MB (usar MP3 128kbps)
- Video: < 50MB (usar MP4 con compresión moderada)

### 3. Token Management

```json
{
  "max_tokens": 500,
  "temperature": 0.3
}
```

- `max_tokens`: Limita longitud de respuesta
- `temperature`: 0.0 para deterministico, 1.0+ para creativo

---

## 7. SDKs y Librerías

### Python

```python
import requests
import json

class LLMGateway:
    def __init__(self, base_url="http://localhost:8765"):
        self.base_url = base_url
    
    def chat(self, task, privacy_mode, messages, **kwargs):
        data = {
            "task": task,
            "privacy_mode": privacy_mode,
            "messages": json.dumps(messages),
            **kwargs
        }
        
        response = requests.post(
            f"{self.base.url}/v1/chat/completions",
            data=data
        )
        return response.json()
    
    def embed_text(self, text, normalize=True):
        response = requests.post(
            f"{self.base_url}/v1/embeddings/text",
            json={"text": text, "normalize": normalize}
        )
        return response.json()

# Uso
client = LLMGateway()
result = client.chat("chat", "strict", [
    {"role": "user", "content": "Hola"}
])
```

---

## 8. Versioning

**Versión actual:** v1

Todos los endpoints incluyen `/v1/` en su path para versionado futuro.

---

## 9. Support

- **Documentación:** http://localhost:8765/docs
- **Issues:** GitHub Issues
- **Email:** [tu-email]

---

## 10. Changelog

### v1.0.0 (2026-01-05)
- ✅ Chat completions multimodal
- ✅ Google File API para archivos grandes
- ✅ Embeddings multimodales (texto, imagen, audio)
- ✅ Routing inteligente local/cloud
- ✅ Optimizaciones RTX 4090
