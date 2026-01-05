"""
Router de Chat Completions Multimodal
Endpoint principal que acepta archivos multimedia directamente
"""
import logging
import time
import uuid
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from schemas.requests import ChatCompletionRequest, ChatMessage
from schemas.responses import ChatCompletionResponse
from services.router import model_router
from services.llm_client import call_llm
from services.file_processor import process_uploaded_file, prepare_multimodal_content

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    # Parámetros obligatorios del form
    task: str = Form(..., description="Tipo de tarea: chat, vision, ocr, embedding"),
    privacy_mode: str = Form(..., description="Modo de privacidad: strict o flexible"),
    messages: str = Form(..., description="JSON string con lista de mensajes"),
    
    # Parámetros opcionales del form
    model: Optional[str] = Form(None, description="Override del modelo"),
    temperature: Optional[float] = Form(0.7, description="Temperatura (0.0-2.0)"),
    max_tokens: Optional[int] = Form(None, description="Máximo de tokens"),
    stream: Optional[bool] = Form(False, description="Streaming habilitado"),
    top_p: Optional[float] = Form(None, description="Top-p sampling"),
    
    # Archivos multimedia opcionales
    files: Optional[List[UploadFile]] = File(default=None, description="Archivos multimedia adjuntos")
):
    """
    Endpoint de chat completions con soporte multimodal
    
    **Formato Multipart/Form-Data:**
    - task: "chat" | "vision" | "ocr"
    - privacy_mode: "strict" | "flexible"
    - messages: JSON string con mensajes
    - files: Archivos multimedia (opcional)
    
    **Ejemplo de messages:**
    ```json
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "¿Qué ves en esta imagen?"},
                {"type": "image", "file_index": 0}
            ]
        }
    ]
    ```
    """
    try:
        # 1. Validar task y privacy_mode primero (antes de procesar archivos)
        if task not in ["chat", "vision", "ocr"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Task inválido: {task}. Debe ser: chat, vision, ocr"
            )
        
        if privacy_mode not in ["strict", "flexible"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Privacy mode inválido: {privacy_mode}. Debe ser: strict, flexible"
            )
        
        # 2. Parsear messages JSON
        try:
            messages_data = json.loads(messages)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error parseando messages JSON: {str(e)}"
            )
        
        # 3. Procesar archivos si existen
        files_metadata = []
        if files:
            logger.info(f"📎 Procesando {len(files)} archivo(s) adjunto(s)")
            for idx, file in enumerate(files):
                try:
                    # Determinar tipo de archivo según content-type
                    file_type = "image"  # Default
                    if file.content_type:
                        if file.content_type.startswith("image/"):
                            file_type = "image"
                        elif file.content_type.startswith("audio/"):
                            file_type = "audio"
                        elif file.content_type.startswith("application/"):
                            file_type = "document"
                    
                    file_meta = await process_uploaded_file(file, file_type)
                    files_metadata.append(file_meta)
                    logger.info(
                        f"  [{idx}] {file_meta['filename']} "
                        f"({file_meta['size'] / 1024:.1f}KB, {file_meta['content_type']})"
                    )
                except ValueError as ve:
                    # Error específico de validación de archivo
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Error en archivo {idx}: {str(ve)}"
                    )
        
        # 4. Preparar mensajes con archivos
        try:
            prepared_messages = prepare_multimodal_content(
                messages_data, 
                files_metadata,
                privacy_mode=privacy_mode  # Pasar privacy_mode para decisión de File API
            )
        except NotImplementedError as e:
            # Archivo grande con privacy_mode=strict (chunking no implementado)
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=str(e)
            )
        
        # 5. Seleccionar modelo usando el router
        selected_model = model_router.select_model(
            task=task,
            privacy_mode=privacy_mode,
            override_model=model
        )
        
        logger.info(f"🎯 Modelo seleccionado: {selected_model}")
        logger.info(f"📨 Enviando {len(prepared_messages)} mensaje(s) al LLM")
        
        # 6. Llamar al LLM
        llm_response = await call_llm(
            model=selected_model,
            messages=prepared_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            top_p=top_p
        )
        
        # 7. Convertir respuesta de LiteLLM a nuestro formato
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
        
        logger.info(f"✅ Respuesta generada exitosamente")
        return response_dict
        
    except ValueError as e:
        # Errores de validación
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
