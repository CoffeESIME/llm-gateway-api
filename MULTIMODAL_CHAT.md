# Chat Completions Multimodal - Documentación

## 📚 Descripción

Endpoint unificado de chat completions que acepta archivos multimedia directamente en la petición usando `multipart/form-data`. 

**Características:**
- ✅ Acepta archivos directamente (no URLs)
- ✅ Soporta imágenes, audio y documentos
- ✅ Routing inteligente entre modelos locales y cloud
- ✅ Compatible con formato OpenAI (extendido)

---

## 📊 Límites de Archivos

La API maneja archivos de forma inteligente según su tamaño:

| Tamaño | Estrategia | Privacy Mode Requerido | Max Size |
|--------|------------|------------------------|----------|
| **< 5MB** | Base64 data URI | Cualquiera | 5MB |
| **>= 5MB** | Google File API | `flexible` solamente | 100MB |

**Google File API:**
- Archivos >= 5MB se suben temporalmente a servidores de Google
- Se obtiene una URI que se pasa a Gemini
- Google auto-elimina archivos después de 48 horas
- Solo funciona con `privacy_mode=flexible` (Gemini)

**Archivos grandes con `privacy_mode=strict`:**
- No soportado actualmente (retorna error 501)
- En roadmap: chunking local con ffmpeg para video/audio

---

## 🚀 Uso Básico

### Endpoint

```
POST /v1/chat/completions
Content-Type: multipart/form-data
```

### Parámetros del Form

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `task` | string | ✅ | Tipo de tarea: `chat`, `vision`, `ocr`, `embedding` |
| `privacy_mode` | string | ✅ | Modo de privacidad: `strict` (local) o `flexible` (cloud) |
| `messages` | JSON string | ✅ | Array de mensajes en formato JSON |
| `files` | File[] | ❌ | Archivos multimedia adjuntos (opcional) |
| `model` | string | ❌ | Override manual del modelo |
| `temperature` | float | ❌ | Temperatura (0.0-2.0), default: 0.7 |
| `max_tokens` | int | ❌ | Máximo de tokens a generar |
| `stream` | bool | ❌ | Habilitar streaming, default: false |
| `top_p` | float | ❌ | Top-p sampling (0.0-1.0) |

---

## 📝 Formato de Mensajes

### Chat Simple (Sin Archivos)

```json
[
    {
        "role": "user",
        "content": "¿Cuál es la capital de Francia?"
    }
]
```

### Multimodal (Con Archivos)

```json
[
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "¿Qué ves en esta imagen?"
            },
            {
                "type": "image",
                "file_index": 0
            }
        ]
    }
]
```

**Tipos de contenido:**
- `text`: Texto plano
- `image`: Referencia a imagen adjunta
- `audio`: Referencia a audio adjunto
- `document`: Referencia a documento adjunto

**file_index:** Índice del archivo en el array de archivos (0-based)

---

## 💡 Ejemplos de Uso

### Ejemplo 1: Chat Simple con curl

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=chat' \
  -F 'privacy_mode=strict' \
  -F 'messages=[{"role":"user","content":"Explica qué es un embedding"}]' \
  -F 'temperature=0.7'
```

### Ejemplo 2: Vision con Imagen

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=vision' \
  -F 'privacy_mode=flexible' \
  -F 'messages=[{"role":"user","content":[{"type":"text","text":"Describe esta imagen"},{"type":"image","file_index":0}]}]' \
  -F 'files=@imagen.jpg'
```

### Ejemplo 3: OCR de Documento

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=ocr' \
  -F 'privacy_mode=strict' \
  -F 'messages=[{"role":"user","content":[{"type":"text","text":"Extrae el texto"},{"type":"image","file_index":0}]}]' \
  -F 'files=@documento.png' \
  -F 'temperature=0.0'
```

### Ejemplo 4: Conversación con Contexto

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=chat' \
  -F 'privacy_mode=strict' \
  -F 'messages=[
    {"role":"system","content":"Eres un experto en inteligencia artificial"},
    {"role":"user","content":"¿Qué es RAG?"},
    {"role":"assistant","content":"RAG significa Retrieval Augmented Generation..."},
    {"role":"user","content":"Dame un ejemplo práctico"}
  ]' \
  -F 'temperature=0.7' \
  -F 'max_tokens=200'
```

---

## 🐍 Uso desde Python

### Ejemplo 1: Chat Simple

```python
import requests
import json

url = "http://localhost:8765/v1/chat/completions"

messages = [
    {"role": "user", "content": "¿Qué es un vector embedding?"}
]

data = {
    "task": "chat",
    "privacy_mode": "strict",
    "messages": json.dumps(messages),
    "temperature": "0.7"
}

response = requests.post(url, data=data)
result = response.json()

print(result["choices"][0]["message"]["content"])
```

### Ejemplo 2: Vision con Imagen

```python
import requests
import json

url = "http://localhost:8765/v1/chat/completions"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe lo que ves"},
            {"type": "image", "file_index": 0}
        ]
    }
]

data = {
    "task": "vision",
    "privacy_mode": "flexible",
    "messages": json.dumps(messages),
    "temperature": "0.7"
}

# Adjuntar imagen
with open("foto.jpg", "rb") as f:
    files = {"files": ("foto.jpg", f, "image/jpeg")}
    response = requests.post(url, data=data, files=files)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Ejemplo 3: Múltiples Archivos

```python
import requests
import json

url = "http://localhost:8765/v1/chat/completions"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compara estas dos imágenes"},
            {"type": "image", "file_index": 0},
            {"type": "image", "file_index": 1}
        ]
    }
]

data = {
    "task": "vision",
    "privacy_mode": "flexible",
    "messages": json.dumps(messages)
}

# Adjuntar múltiples archivos
files = [
    ("files", ("imagen1.jpg", open("imagen1.jpg", "rb"), "image/jpeg")),
    ("files", ("imagen2.jpg", open("imagen2.jpg", "rb"), "image/jpeg"))
]

response = requests.post(url, data=data, files=files)
result = response.json()
print(result["choices"][0]["message"]["content"])
```

---

## 📊 Formato de Respuesta

```json
{
    "id": "chatcmpl-a1b2c3d4",
    "object": "chat.completion",
    "created": 1704398400,
    "model": "ollama/qwen3-vl:8b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "La respuesta del modelo..."
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

---

## ⚠️ Limitaciones y Validaciones

### Tamaños de Archivo
- **Máximo por archivo:** 10MB
- **Formatos soportados:**
  - Imágenes: JPEG, PNG, WEBP, GIF
  - Audio: WAV, MP3, FLAC, OGG
  - Documentos: PDF, TXT, imágenes (para OCR)

### Validaciones
- El `file_index` debe corresponder a un archivo adjunto válido
- El tipo de archivo debe coincidir con el `content-type`
- Todos los parámetros obligatorios deben estar presentes

---

## 🔧 Pruebas

### Ejecutar Tests Automatizados

```bash
python test_multimodal_chat.py
```

### Pruebas con Archivos Reales

```python
# En Python REPL o script
from test_multimodal_chat import *

# Vision con imagen
test_vision_with_image("mi_foto.jpg")

# OCR con documento
test_ocr_with_document("documento.png")
```

---

## 🎯 Casos de Uso

### 1. Análisis de Imágenes
```bash
# Detectar objetos
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=vision' \
  -F 'privacy_mode=flexible' \
  -F 'messages=[{"role":"user","content":[{"type":"text","text":"Lista todos los objetos que ves"},{"type":"image","file_index":0}]}]' \
  -F 'files=@escena.jpg'
```

### 2. Extracción de Texto (OCR)
```bash
# Extraer texto de factura
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=ocr' \
  -F 'privacy_mode=strict' \
  -F 'messages=[{"role":"user","content":[{"type":"text","text":"Extrae: fecha, total y número de factura"},{"type":"image","file_index":0}]}]' \
  -F 'files=@factura.png' \
  -F 'temperature=0.0'
```

### 3. Chat con Privacidad Estricta
```bash
# Datos confidenciales - solo modelos locales
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=chat' \
  -F 'privacy_mode=strict' \
  -F 'messages=[{"role":"user","content":"Resume este reporte confidencial"}]'
```

---

## 🐛 Troubleshooting

### Error: "file_index X inválido"
**Causa:** El índice referenciado no existe en los archivos adjuntos  
**Solución:** Asegúrate de que file_index esté dentro del rango [0, número_de_archivos-1]

### Error: "Archivo demasiado grande"
**Causa:** El archivo excede el límite de 10MB  
**Solución:** Comprime el archivo o divídelo en partes más pequeñas

### Error: "Error parseando messages JSON"
**Causa:** El campo messages no es JSON válido  
**Solución:** Usa `json.dumps()` para convertir el array a string JSON

### Error: "Tipo de archivo no soportado"
**Causa:** El content-type del archivo no está en la lista permitida  
**Solución:** Verifica que el archivo sea JPEG, PNG, WAV, etc.

---

## 📚 Documentación Interactiva

Accede a la documentación Swagger en:
```
http://localhost:8765/docs
```

---

## 🔗 Ver También

- [Documentación de Embeddings](EMBEDDINGS.md)
- [Optimizaciones RTX 4090](RTX4090_OPTIMIZATIONS.md)
- [README Principal](README.md)
