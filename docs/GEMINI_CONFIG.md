# Configuración de Gemini API

## 🔑 Configuración de API Key

Para usar los modelos de Gemini (Google AI), necesitas configurar tu API key en el archivo `.env`.

### 1. Crear/Editar archivo `.env`

```bash
# Copiar el ejemplo si no existe
cp .env.example .env
```

### 2. Agregar tu API Key de Google

Edita el archivo `.env` y agrega tu API key:

```env
# API Keys
GEMINI_API_KEY=tu-api-key-aqui

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Logging
LOG_LEVEL=INFO
```

**¿Dónde obtener la API key?**
1. Ve a [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crea un nuevo proyecto o selecciona uno existente
3. Genera una nueva API key
4. Cópiala y pégala en el archivo `.env`

---

## 🎯 Modelos Configurados

Los modelos de Gemini se usan en modo `privacy_mode=flexible`:

| Tarea | Modo Strict (Local) | Modo Flexible (Cloud) |
|-------|---------------------|----------------------|
| **chat** | `ollama/dolphin-mistral-nemo` | `gemini/gemini-2.0-flash-exp` |
| **vision** | `ollama/qwen3-vl:8b` | `gemini/gemini-2.0-flash-exp` |
| **ocr** | `ollama/deepseek-ocr:3b` | `gemini/gemini-2.0-flash-exp` |

---

## ⚙️ Formato de Modelos en LiteLLM

**IMPORTANTE:** Los modelos de Gemini deben usar el formato correcto:

✅ **Correcto:**
```python
"gemini/gemini-2.0-flash-exp"  # Con prefijo gemini/
```

❌ **Incorrecto:**
```python
"gemini-2.0-flash-exp"  # Sin prefijo - intentará usar Vertex AI
```

**¿Por qué?**
- El prefijo `gemini/` le dice a LiteLLM que use **Google AI API** (con API key)
- Sin el prefijo, LiteLLM asume **Vertex AI** (requiere Google Cloud credentials)

---

## 🧪 Verificar Configuración

### Test 1: Verificar que la API key se carga

```bash
python -c "from config import settings; print(f'API Key configurada: {bool(settings.gemini_api_key)}')"
```

**Resultado esperado:**
```
API Key configurada: True
```

### Test 2: Probar llamada a Gemini

```bash
python test_multimodal_chat.py
```

O usa curl:

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -F 'task=chat' \
  -F 'privacy_mode=flexible' \
  -F 'messages=[{"role":"user","content":"Hola, ¿cómo estás?"}]'
```

**Si funciona correctamente, verás:**
```
✅ Respuesta recibida:
   Modelo: gemini/gemini-2.0-flash-exp
   Respuesta: ¡Hola! Estoy bien, gracias...
```

---

## 🐛 Troubleshooting

### Error: "AuthenticationError"

```
Error de autenticación. Verifica tu GEMINI_API_KEY en .env
```

**Solución:**
1. Verifica que el archivo `.env` existe en la raíz del proyecto
2. Verifica que `GEMINI_API_KEY` está configurado correctamente
3. Asegúrate de que no hay espacios extras: `GEMINI_API_KEY=AIza...` (sin espacios)
4. Reinicia el servidor después de cambiar el `.env`

### Error: "Vertex AI credentials not found"

```
Could not automatically determine credentials for Vertex AI
```

**Causa:** El modelo no tiene el prefijo `gemini/`

**Solución:** Verifica `config.py`:
```python
MODEL_ROUTER = {
    "vision": {
        "flexible": "gemini/gemini-2.0-flash-exp"  # ✅ Correcto
    }
}
```

### Error: "Model not found"

```
Modelo 'gemini/gemini-2.0-flash-exp' no encontrado
```

**Posibles causas:**
1. API key inválida o expirada
2. Modelo no disponible en tu región
3. Nombre del modelo incorrecto

**Soluciones:**
1. Verifica la API key en Google AI Studio
2. Prueba otro modelo: `gemini/gemini-1.5-flash`
3. Revisa los [modelos disponibles](https://ai.google.dev/models)

### Error: "Rate limit exceeded"

```
Rate limit excedido
```

**Solución:**
1. Espera unos minutos antes de reintentar
2. Considera usar modo `strict` (modelos locales)
3. Aumenta el límite de rate en Google AI Studio (si es posible)

---

## 📊 Logs de Depuración

Para ver logs detallados de LiteLLM:

### Opción 1: Modo Verbose en código

Edita `services/llm_client.py`:
```python
litellm.set_verbose = True  # Cambiar a True
```

### Opción 2: Variable de entorno

```bash
export LITELLM_LOG=DEBUG  # Linux/Mac
set LITELLM_LOG=DEBUG     # Windows CMD
$env:LITELLM_LOG="DEBUG"  # Windows PowerShell

python main.py
```

**Logs útiles:**
- `📤 Llamando a modelo: gemini/...` - Confirma el modelo usado
- `GEMINI_API_KEY` encontrada - Confirma que la API key se cargó
- `Request to https://generativelanguage.googleapis.com/...` - Confirma que usa Google AI API

---

## 🔐 Seguridad

### ⚠️ NUNCA Subas tu API Key a GitHub

Asegúrate de que `.env` está en `.gitignore`:

```bash
# Verificar
cat .gitignore | grep .env
```

**Debería mostrar:**
```
.env
*.env
```

### 🔄 Rotar API Keys

Si accidentalmente expones tu API key:
1. Ve a [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Revoca la API key comprometida
3. Genera una nueva
4. Actualiza el archivo `.env`

---

## 📚 Referencias

- [Google AI Studio](https://makersuite.google.com/)
- [Documentación de Google AI](https://ai.google.dev/docs)
- [LiteLLM - Gemini Support](https://docs.litellm.ai/docs/providers/gemini)
- [Modelos disponibles](https://ai.google.dev/models)

---

## ✅ Checklist de Configuración

- [ ] Archivo `.env` creado en la raíz del proyecto
- [ ] `GEMINI_API_KEY` configurado con tu API key
- [ ] Modelos en `config.py` usan formato `gemini/modelo-name`
- [ ] API key válida y activa en Google AI Studio
- [ ] Servidor reiniciado después de cambios en `.env`
- [ ] Test de conexión exitoso
