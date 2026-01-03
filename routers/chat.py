"""
Router de Chat Completions
Endpoint principal compatible con formato OpenAI
"""
import logging
import time
import uuid
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from schemas.requests import ChatCompletionRequest
from schemas.responses import ChatCompletionResponse, Choice, Message, Usage
from services.router import model_router
from services.llm_client import call_llm

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint de chat completions compatible con formato OpenAI
    
    Enruta automáticamente a modelos locales (Ollama) o cloud (Gemini)
    basado en el task y privacy_mode especificados
    """
    try:
        # 1. Seleccionar modelo usando el router
        selected_model = model_router.select_model(
            task=request.task,
            privacy_mode=request.privacy_mode,
            override_model=request.model
        )
        
        # 2. Preparar mensajes
        messages = [msg.dict() for msg in request.messages]
        
        # 3. Llamar al LLM
        llm_response = await call_llm(
            model=selected_model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            top_p=request.top_p
        )
        
        # 4. Convertir respuesta de LiteLLM a nuestro formato
        # LiteLLM ya retorna en formato OpenAI, solo necesitamos asegurar compatibilidad
        response_dict = {
            "id": llm_response.id if hasattr(llm_response, 'id') else f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": llm_response.created if hasattr(llm_response, 'created') else int(time.time()),
            "model": selected_model,
            "choices": [
                {
                    "index": choice.index if hasattr(choice, 'index') else idx,
                    "message": {
                        "role": choice.message.role if hasattr(choice.message, 'role') else "assistant",
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else "stop"
                }
                for idx, choice in enumerate(llm_response.choices)
            ]
        }
        
        # Agregar usage si está disponible
        if hasattr(llm_response, 'usage') and llm_response.usage:
            response_dict["usage"] = {
                "prompt_tokens": llm_response.usage.prompt_tokens,
                "completion_tokens": llm_response.usage.completion_tokens,
                "total_tokens": llm_response.usage.total_tokens
            }
        
        return response_dict
        
    except ValueError as e:
        # Errores de validación de routing
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        # Errores generales
        logger.error(f"Error en chat completions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar la petición: {str(e)}"
        )


@router.get("/models")
async def list_models():
    """
    Lista los modelos y configuraciones disponibles
    """
    return {
        "available_tasks": model_router.get_available_tasks(),
        "model_configuration": model_router.model_map
    }
