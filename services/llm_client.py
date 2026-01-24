"""
Cliente LiteLLM para interactuar con modelos locales y cloud
"""
import logging
import traceback
import os
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


async def call_llm(
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
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
        
        # Llamar a LiteLLM
        logger.debug("🔄 Enviando request a LiteLLM...")
        response = completion(**params)
        
        # Log de respuesta exitosa
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content_preview = response.choices[0].message.content[:200]
            logger.info(f"✅ Respuesta recibida ({len(content_preview)} chars): {content_preview}...")
            
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

