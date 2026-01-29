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
        
        # Preparar parámetros
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Si se requiere respuesta JSON, agregar response_format
        if json_response:
            params["response_format"] = {"type": "json_object"}
            logger.debug("📋 Solicitando response_format: json_object")
        
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
            
            content_preview = response.choices[0].message.content[:200]
            logger.info(f"✅ Respuesta recibida ({len(response.choices[0].message.content)} chars): {content_preview}...")
            
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

