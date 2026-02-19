"""
Cliente LiteLLM para interactuar con modelos locales y cloud
"""
import logging
import traceback
import os
import re
from typing import Dict, Any, Optional
import litellm
from litellm import completion

from config import settings

logger = logging.getLogger(__name__)

# Configurar LiteLLM
litellm.set_verbose = False  # Desactivar verbose para logs limpios

# Configurar API keys
if settings.gemini_api_key:
    os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
    logger.debug("✅ GEMINI_API_KEY configurada")

# Configurar base URL de Ollama
os.environ["OLLAMA_API_BASE"] = settings.ollama_base_url
logger.debug(f"✅ OLLAMA_API_BASE: {settings.ollama_base_url}")


def _prepare_ocr_messages(messages: list) -> list:
    """
    Prepara mensajes para tareas de OCR con modelos de visión.
    
    Usa un prompt estructurado que:
    - Extrae todo el texto de forma literal
    - Preserva el formato original (saltos de línea, estrofas)
    - Mantiene la ortografía española (acentos, ñ)
    - Devuelve SOLO el texto sin conversación adicional
    """
    # Prompt OCR optimizado para modelos inteligentes (qwen3-vl, minicpm-v)
    OCR_INSTRUCTION = """Analyze the text in this image.
1. Extract all text verbatim.
2. Preserve the original formatting (line breaks, stanzas).
3. Maintain strict Spanish orthography (accents, ñ).
4. Output ONLY the text, no conversational filler."""
    
    # Buscar las imágenes en los mensajes
    image_parts = []
    
    for msg in messages:
        content = msg.get('content', '')
        
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'image_url':
                    image_parts.append(part)
    
    if not image_parts:
        logger.warning("⚠️ No se encontraron imágenes para OCR")
        return messages  # Fallback al formato original
    
    # Construir mensaje: [imágenes] + instrucción OCR
    prepared_content = image_parts + [{"type": "text", "text": OCR_INSTRUCTION}]
    
    logger.info(f"🎯 OCR: {len(image_parts)} imagen(es) preparadas")
    
    return [{"role": "user", "content": prepared_content}]


def clean_json_response(response: str) -> str:
    """
    Limpia respuestas JSON envueltas en markdown code blocks.
    
    Los modelos cloud (Gemini, OpenAI) a veces devuelven JSON envuelto en:
    ```json
    {"key": "value"}
    ```
    
    Esta función extrae el JSON limpio.
    
    Args:
        response: String de respuesta del modelo
        
    Returns:
        JSON limpio sin markdown code blocks
    """
    # Log del contenido original para debug
    print(f"🔍 [clean_json] Input length: {len(response) if response else 0}")
    print(f"🔍 [clean_json] Input preview: {response[:300] if response else 'EMPTY/NONE'}...")
    
    if not response:
        print("⚠️ [clean_json] Response is empty or None!")
        return response
    
    # Patrón para detectar ```json ... ``` o ``` ... ```
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(pattern, response)
    
    if match:
        extracted = match.group(1).strip()
        print(f"🧹 [clean_json] Found markdown code block!")
        print(f"🧹 [clean_json] Extracted length: {len(extracted)}")
        print(f"🧹 [clean_json] Extracted preview: {extracted[:300] if extracted else 'EMPTY'}...")
        
        if not extracted:
            print("⚠️ [clean_json] Extracted content is EMPTY after stripping markdown!")
        
        return extracted
    
    # No se encontró markdown code block, devolver stripped
    result = response.strip()
    print(f"🔍 [clean_json] No markdown found, returning stripped ({len(result)} chars)")
    return result


async def call_llm(
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    json_response: bool = True,
    task: Optional[str] = None,
    top_p: float = 0.9,
    **kwargs
) -> Dict[str, Any]:
    """
    Llama al LLM usando LiteLLM
    
    Args:
        model: Nombre del modelo (formato: provider/model-name)
        messages: Lista de mensajes en formato OpenAI
        temperature: Temperatura de generación
        max_tokens: Máximo tokens a generar
        stream: Si se desea streaming
        json_response: Si es True (default), solicita respuesta JSON y limpia
                      markdown code blocks. Si es False, devuelve respuesta raw.
        task: Tipo de tarea (chat, vision, ocr) para manejo especial
        **kwargs: Parámetros adicionales para el modelo
    
    Returns:
        Dict con la respuesta del modelo
    
    Raises:
        Exception: Si hay error en la llamada al modelo
    """
    try:
        logger.info(f"📤 Llamando a modelo: {model}")
        logger.debug(f"   Temperature: {temperature}")
        logger.debug(f"   Max tokens: {max_tokens}")
        logger.debug(f"   Stream: {stream}")
        logger.debug(f"   JSON response: {json_response}")
        logger.debug(f"   Messages count: {len(messages)}")
        
        # Log del contenido de mensajes (solo primeros 200 chars)
        for idx, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if isinstance(content, str):
                content_preview = content[:200] + ('...' if len(content) > 200 else '')
                logger.debug(f"   Message {idx} [{role}]: {content_preview}")
            else:
                logger.debug(f"   Message {idx} [{role}]: <multimodal content>")
        
        # ========================================
        # MANEJO ESPECIAL: Tareas de OCR
        # ========================================
        # Solo aplicar lógica OCR cuando task == "ocr"
        is_ocr_task = task == "ocr"
        
        if is_ocr_task:
            logger.info("🔧 Tarea OCR detectada, preparando mensajes y temperatura...")
            messages = _prepare_ocr_messages(messages)
            temperature = 0.0  # OCR debe ser determinístico
            json_response = False  # OCR devuelve texto plano
            logger.debug(f"   Mensajes reformateados: {len(messages)} mensaje(s), temp=0.0")
        
        # Preparar parámetros
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            **kwargs
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Si se requiere respuesta JSON, agregar response_format
        if json_response:
            params["response_format"] = {"type": "json_object"}
            logger.debug("📋 Solicitando response_format: json_object")
        
        # ============== DEBUG PRINTS ==============
        print("\n" + "="*80)
        print("🚀 [DEBUG] LLAMADA A LLM - PROMPT ENVIADO")
        print("="*80)
        print(f"📌 Modelo: {model}")
        print(f"🌡️  Temperature: {temperature}")
        print(f"📊 Max tokens: {max_tokens}")
        print(f"📑 JSON response mode: {json_response}")
        print(f"🔢 Número de mensajes: {len(messages)}")
        print("-"*80)
        print("📝 MENSAJES COMPLETOS:")
        for idx, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            print(f"\n--- Mensaje {idx} [{role.upper()}] ---")
            if isinstance(content, str):
                print(content)
            elif isinstance(content, list):
                # Contenido multimodal
                for part_idx, part in enumerate(content):
                    if isinstance(part, dict):
                        if part.get('type') == 'text':
                            print(f"  [TEXT PART {part_idx}]: {part.get('text', '')}")
                        elif part.get('type') == 'image_url':
                            img_url = part.get('image_url', {}).get('url', '')
                            print(f"  [IMAGE PART {part_idx}]: {img_url[:100]}..." if len(img_url) > 100 else f"  [IMAGE PART {part_idx}]: {img_url}")
                        else:
                            print(f"  [OTHER PART {part_idx}]: {part}")
                    else:
                        print(f"  [PART {part_idx}]: {part}")
            else:
                print(f"  <contenido tipo: {type(content)}>")
        print("-"*80)
        print(f"📦 Params adicionales (kwargs): {kwargs}")
        print("="*80 + "\n")
        # ========== FIN DEBUG PRINTS ==========
        
        # Llamar a LiteLLM
        logger.debug("🔄 Enviando request a LiteLLM...")
        response = completion(**params)
        
        # Log de respuesta exitosa y limpiar si es necesario
        if hasattr(response, 'choices') and len(response.choices) > 0:
            original_content = response.choices[0].message.content
            
            # Si json_response está activo, limpiar markdown code blocks
            if json_response and original_content:
                cleaned_content = clean_json_response(original_content)
                response.choices[0].message.content = cleaned_content
                
                if cleaned_content != original_content:
                    logger.debug("🧹 Respuesta JSON limpiada de markdown code blocks")
            
            final_content = response.choices[0].message.content
            content_len = len(final_content) if final_content else 0
            content_preview = final_content[:200] if final_content else ""
            logger.info(f"✅ Respuesta recibida ({content_len} chars): {content_preview}...")
            
            # Verificar que la respuesta no esté vacía
            if not final_content or content_len == 0:
                logger.error(f"❌ El modelo {model} devolvió una respuesta vacía (0 chars)")
                raise Exception(
                    f"El modelo '{model}' devolvió una respuesta vacía. "
                    "Esto puede ocurrir si el modelo no soporta el tipo de contenido enviado "
                    "o si hay un problema con los parámetros de la petición."
                )
            
            # Log de usage si está disponible
            if hasattr(response, 'usage') and response.usage:
                logger.debug(f"📊 Usage: {response.usage.total_tokens} tokens total")
        
        return response
        
    except litellm.exceptions.Timeout as e:
        logger.error(f"⏱️ Timeout al llamar modelo {model}")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Stack trace: {traceback.format_exc()}")
        raise Exception(f"Timeout al llamar al modelo: {str(e)}")
    
    except litellm.exceptions.RateLimitError as e:
        logger.error(f"🚫 Rate limit excedido para {model}")
        logger.error(f"   Error: {str(e)}")
        raise Exception(f"Rate limit excedido: {str(e)}")
    
    except litellm.exceptions.BadRequestError as e:
        logger.error(f"❌ Bad request para {model}")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   Params: model={model}, temp={temperature}, max_tokens={max_tokens}")
        logger.error(f"   Stack trace: {traceback.format_exc()}")
        raise Exception(f"Error en la petición: {str(e)}")
    
    except litellm.exceptions.NotFoundError as e:
        logger.error(f"🔍 Modelo no encontrado: {model}")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   ¿Ollama está corriendo? Check: {settings.ollama_base_url}")
        raise Exception(
            f"Modelo '{model}' no encontrado. "
            f"Verifica que Ollama esté corriendo y el modelo descargado."
        )
    
    except litellm.exceptions.AuthenticationError as e:
        logger.error(f"🔑 Error de autenticación para {model}")
        logger.error(f"   Error: {str(e)}")
        logger.error(f"   GEMINI_API_KEY configurada: {bool(settings.gemini_api_key)}")
        raise Exception(
            f"Error de autenticación. Verifica tu GEMINI_API_KEY en .env"
        )
    
    except Exception as e:
        logger.error(f"💥 ERROR INESPERADO al llamar {model}")
        logger.error(f"   Tipo: {type(e).__name__}")
        logger.error(f"   Mensaje: {str(e)}")
        logger.error(f"   Stack trace completo:")
        logger.error(f"{traceback.format_exc()}")
        raise Exception(f"Error al llamar al modelo: {str(e)}")

