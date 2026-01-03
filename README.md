# LLM Gateway API

API Gateway inteligente que enruta automáticamente peticiones entre modelos locales (Ollama) y modelos en la nube (Google Gemini) basándose en el parámetro de privacidad del usuario.

## 🎯 Características

- **Routing Inteligente**: Selección automática de modelo basada en tipo de tarea y modo de privacidad
- **Multi-Modal**: Soporte para texto, visión, OCR y embeddings
- **Compatible OpenAI**: Formato de API compatible con OpenAI Chat Completions
- **Local + Cloud**: Usa modelos locales Ollama para privacidad estricta, Gemini para flexibilidad
- **FastAPI**: API moderna con documentación automática (Swagger)

## 📋 Requisitos Previos

1. **Python 3.10+**
2. **Ollama** instalado y corriendo con los siguientes modelos:
   - `CognitiveComputations/dolphin-mistral-nemo:latest`
   - `qwen3-vl:8b`
   - `deepseek-ocr:3b`
   - `nomic-embed-text:latest`
3. **Google Gemini API Key** (para modo flexible)

## 🚀 Instalación

### 1. Clonar e instalar dependencias

```bash
# Navegar al directorio
cd llm-endpoints

# Crear entorno virtual (recomendado)
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
# Copiar el archivo de ejemplo
copy .env.example .env

# Editar .env con tu editor y agregar tu GEMINI_API_KEY
# Asegúrate de reemplazar 'your_gemini_api_key_here' con tu API key real
```

> [!IMPORTANT]
> **Debes crear el archivo `.env`** copiando `.env.example` y agregando tu `GEMINI_API_KEY` real.
> Sin esta API key, solo podrás usar `privacy_mode: "strict"` (modelos locales).

### 3. Verificar Ollama

```bash
# Verificar que Ollama está corriendo
ollama list

# Debería mostrar los modelos instalados
# Si faltan modelos, descargarlos:
# ollama pull CognitiveComputations/dolphin-mistral-nemo:latest
# ollama pull qwen3-vl:8b
# ollama pull deepseek-ocr:3b
# ollama pull nomic-embed-text:latest
```

### 4. Iniciar el servidor

```bash
# Opción 1: Usar uvicorn directamente
uvicorn main:app --reload --port 8765

# Opción 2: Ejecutar el script main.py
python main.py
```

El servidor estará disponible en: `http://localhost:8765`

## 📚 Documentación Interactiva

Una vez iniciado el servidor:
- **Swagger UI**: http://localhost:8765/docs
- **ReDoc**: http://localhost:8765/redoc

## 🔧 Uso

### Estructura de la Petición

```json
{
  "task": "chat | vision | ocr | embedding",
  "privacy_mode": "strict | flexible",
  "messages": [
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.7,
  "max_tokens": 500
}
```

**Parámetros principales:**
- `task`: Tipo de tarea a realizar
- `privacy_mode`: 
  - `strict`: Usa modelos locales (Ollama)
  - `flexible`: Usa modelos cloud (Gemini)

### Ejemplo 1: Chat Privado (Local)

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "task": "chat",
  "privacy_mode": "strict",
  "messages": [{"role": "user", "content": "Resume este texto confidencial..."}],
  "temperature": 0.7,
  "max_tokens": 500
}'
```

**Modelo usado**: `ollama/CognitiveComputations/dolphin-mistral-nemo:latest`

### Ejemplo 2: Análisis de Imagen (Cloud)

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "task": "vision",
  "privacy_mode": "flexible",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "¿Qué lugar es este?"},
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
      ]
    }
  ]
}'
```

**Modelo usado**: `gemini/gemini-2.5-pro`

### Ejemplo 3: OCR Local

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "task": "ocr",
  "privacy_mode": "strict",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Extrae el texto de esta imagen"},
        {"type": "image_url", "image_url": {"url": "URL_DE_TU_IMAGEN"}}
      ]
    }
  ]
}'
```

**Modelo usado**: `ollama/deepseek-ocr:3b`

## 🗺️ Routing de Modelos

| Task | Privacy: Strict (Local) | Privacy: Flexible (Cloud) |
|------|------------------------|---------------------------|
| **chat** | `ollama/dolphin-mistral-nemo:latest` | `gemini/gemini-2.5-flash` |
| **vision** | `ollama/qwen3-vl:8b` | `gemini/gemini-2.5-pro` |
| **ocr** | `ollama/deepseek-ocr:3b` | `gemini/gemini-2.5-flash` |
| **embedding** | `ollama/nomic-embed-text:latest` | `ollama/nomic-embed-text:latest` |

## 📁 Estructura del Proyecto

```
llm-endpoints/
├── main.py                 # Aplicación FastAPI principal
├── config.py              # Configuración y MODEL_ROUTER
├── requirements.txt       # Dependencias
├── .env.example          # Template de variables de entorno
├── routers/
│   ├── __init__.py
│   └── chat.py           # Endpoint de chat completions
├── schemas/
│   ├── __init__.py
│   ├── requests.py       # Pydantic schemas de request
│   └── responses.py      # Pydantic schemas de response
└── services/
    ├── __init__.py
    ├── router.py         # Lógica de routing de modelos
    └── llm_client.py     # Cliente LiteLLM
```

## 🔍 Health Check

```bash
curl http://localhost:8765/health
```

Respuesta esperada:
```json
{"status": "ok"}
```

## 📊 Listar Modelos Disponibles

```bash
curl http://localhost:8765/v1/models
```

## 🐛 Troubleshooting

### Error: "Modelo no encontrado"
- Verificar que Ollama está corriendo: `ollama list`
- Descargar el modelo faltante: `ollama pull <modelo>`

### Error: "Error de autenticación"
- Verificar que `GEMINI_API_KEY` está configurada en `.env`
- Verificar que la API key es válida

### Error: "Connection refused"
- Verificar que Ollama está corriendo en `http://localhost:11434`
- Cambiar `OLLAMA_BASE_URL` en `.env` si es necesario

## 📝 Licencia

MIT

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor abre un issue o pull request.
