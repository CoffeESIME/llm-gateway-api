"""
Cliente LiteLLM para interactuar con modelos locales y cloud
"""
import logging
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

# Configurar base URL de Ollama
os.environ["OLLAMA_API_BASE"] = settings.ollama_base_url


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
        logger.debug(f"Messages: {messages}")
        
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
        response = completion(**params)
        
        # Log de respuesta exitosa
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content_preview = response.choices[0].message.content[:100]
            logger.info(f"✅ Respuesta recibida: {content_preview}...")
        
        return response
        
    except litellm.exceptions.Timeout as e:
        logger.error(f"⏱️ Timeout al llamar modelo {model}: {str(e)}")
        raise Exception(f"Timeout al llamar al modelo: {str(e)}")
    
    except litellm.exceptions.RateLimitError as e:
        logger.error(f"🚫 Rate limit excedido para {model}: {str(e)}")
        raise Exception(f"Rate limit excedido: {str(e)}")
    
    except litellm.exceptions.BadRequestError as e:
        logger.error(f"❌ Bad request para {model}: {str(e)}")
        raise Exception(f"Error en la petición: {str(e)}")
    
    except litellm.exceptions.NotFoundError as e:
        logger.error(f"🔍 Modelo no encontrado {model}: {str(e)}")
        raise Exception(
            f"Modelo '{model}' no encontrado. "
            f"Verifica que Ollama esté corriendo y el modelo descargado."
        )
    
    except litellm.exceptions.AuthenticationError as e:
        logger.error(f"🔑 Error de autenticación para {model}: {str(e)}")
        raise Exception(
            f"Error de autenticación. Verifica tu GEMINI_API_KEY en .env"
        )
    
    except Exception as e:
        logger.error(f"💥 Error inesperado al llamar {model}: {str(e)}")
        raise Exception(f"Error al llamar al modelo: {str(e)}")
