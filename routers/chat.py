"""
Router de Chat Completions Multimodal
Endpoint principal que acepta archivos multimedia directamente
"""
import logging
import traceback
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
    top_p: Optional[float] = Form(1.0, description="Top-p sampling (0.0-1.0, default: 1.0)"),
    
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
    request_id = uuid.uuid4().hex[:8]
    start_time = time.time()
    
    # Log de inicio de request
    logger.info(f"\n{'='*60}")
    logger.info(f"🔷 [Request {request_id}] Nueva petición de chat completions")
    logger.info(f"   Task: {task}")
    logger.info(f"   Privacy Mode: {privacy_mode}")
    logger.info(f"   Temperature: {temperature}")
    logger.info(f"   Max Tokens: {max_tokens}")
    logger.info(f"   Model Override: {model}")
    logger.info(f"   Files Attached: {len(files) if files else 0}")
    
    try:
        # 1. Validar task y privacy_mode primero (antes de procesar archivos)
        logger.debug(f"[{request_id}] Validando parámetros...")
        if task not in ["chat", "vision", "ocr"]:
            logger.error(f"[{request_id}] ❌ Task inválido: {task}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Task inválido: {task}. Debe ser: chat, vision, ocr"
            )
        
        if privacy_mode not in ["strict", "flexible"]:
            logger.error(f"[{request_id}] ❌ Privacy mode inválido: {privacy_mode}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Privacy mode inválido: {privacy_mode}. Debe ser: strict, flexible"
            )
        
        # 2. Parsear messages JSON
        logger.debug(f"[{request_id}] Parseando messages JSON...")
        try:
            messages_data = json.loads(messages)
            logger.debug(f"[{request_id}] Messages parseados: {len(messages_data)} mensaje(s)")
        except json.JSONDecodeError as e:
            logger.error(f"[{request_id}] ❌ Error parseando JSON: {str(e)}")
            logger.error(f"[{request_id}] Messages recibidos: {messages[:200]}...")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error parseando messages JSON: {str(e)}"
            )
        
        # 3. Procesar archivos si existen
        files_metadata = []
        if files:
            logger.info(f"[{request_id}] 📎 Procesando {len(files)} archivo(s) adjunto(s)")
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
                    
                    logger.debug(f"[{request_id}] Procesando archivo {idx}: {file.filename} ({file_type})")
                    file_meta = await process_uploaded_file(file, file_type)
                    files_metadata.append(file_meta)
                    logger.info(
                        f"[{request_id}]   [{idx}] {file_meta['filename']} "
                        f"({file_meta['size'] / 1024:.1f}KB, {file_meta['content_type']})"
                    )
                except ValueError as ve:
                    # Error específico de validación de archivo
                    logger.error(f"[{request_id}] ❌ Error validando archivo {idx}: {str(ve)}")
                    logger.error(f"[{request_id}] Filename: {file.filename}, Content-Type: {file.content_type}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Error en archivo {idx}: {str(ve)}"
                    )
                except Exception as e:
                    logger.error(f"[{request_id}] ❌ Error inesperado procesando archivo {idx}:")
                    logger.error(f"[{request_id}] {traceback.format_exc()}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Error procesando archivo {idx}: {str(e)}"
                    )
        
        # 4. Preparar mensajes con archivos
        logger.debug(f"[{request_id}] Preparando contenido multimodal...")
        try:
            prepared_messages = prepare_multimodal_content(
                messages_data, 
                files_metadata,
                privacy_mode=privacy_mode  # Pasar privacy_mode para decisión de File API
            )
            logger.debug(f"[{request_id}] ✅ Mensajes preparados exitosamente")
            
            # DEBUG: Detectar si hay contenido de imagen en los mensajes
            has_image_content = False
            for msg in prepared_messages:
                content = msg.get('content', '')
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'image_url':
                            has_image_content = True
                            break
                if has_image_content:
                    break
            
            print(f"\n🖼️ [DEBUG] Has image content in messages: {has_image_content}")
            print(f"🖼️ [DEBUG] Task: {task}, Privacy: {privacy_mode}")
            
            # Validación: Si hay imágenes pero el task es "chat" con modelo local, advertir
            if has_image_content and task == "chat" and privacy_mode == "strict":
                logger.warning(f"[{request_id}] ⚠️ Mensajes contienen imágenes pero task='chat' con privacy='strict'")
                logger.warning(f"[{request_id}] El modelo local de chat (gpt-oss) NO soporta imágenes!")
                logger.warning(f"[{request_id}] Cambia task='vision' o privacy_mode='flexible'")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "Los mensajes contienen imágenes pero el task='chat' con privacy_mode='strict' "
                        "usa un modelo de solo texto (gpt-oss) que no soporta imágenes. "
                        "Opciones: (1) Usa task='vision' para análisis de imágenes, o "
                        "(2) Usa privacy_mode='flexible' para usar Gemini que soporta visión."
                    )
                )
            
        except NotImplementedError as e:
            # Archivo grande con privacy_mode=strict (chunking no implementado)
            logger.warning(f"[{request_id}] ⚠️ Funcionalidad no implementada: {str(e)[:100]}...")
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Error preparando mensajes multimodales:")
            logger.error(f"[{request_id}] {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error preparando mensajes: {str(e)}"
            )
        
        # 5. Seleccionar modelo usando el router
        logger.debug(f"[{request_id}] Seleccionando modelo...")
        try:
            selected_model = model_router.select_model(
                task=task,
                privacy_mode=privacy_mode,
                override_model=model
            )
            logger.info(f"[{request_id}] 🎯 Modelo seleccionado: {selected_model}")
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Error seleccionando modelo:")
            logger.error(f"[{request_id}] {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error seleccionando modelo: {str(e)}"
            )
        
        logger.info(f"[{request_id}] 📨 Enviando {len(prepared_messages)} mensaje(s) al LLM")
        
        # 6. Llamar al LLM
        # NOTA: Los modelos locales (Ollama) NO soportan response_format=json_object
        # Esto aplica a TODOS los tasks cuando privacy_mode=strict (modelos locales)
        # Solo usamos json_response=True para modelos cloud (flexible) en task chat
        is_local_model = privacy_mode == "strict"
        should_use_json_response = (task == "chat") and (not is_local_model)
        
        # DEBUG: Print para diagnosticar json_response
        print(f"\n🔍 [DEBUG] json_response decision:")
        print(f"   - Task: {task}")
        print(f"   - Privacy mode: {privacy_mode}")
        print(f"   - Is local model: {is_local_model}")
        print(f"   - should_use_json_response: {should_use_json_response}")
        
        try:
            llm_response = await call_llm(
                model=selected_model,
                messages=prepared_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                json_response=should_use_json_response,
                task=task,
                top_p=top_p
            )
        except Exception as e:
            # El error ya está loggeado en llm_client.py
            logger.error(f"[{request_id}] ❌ Error en llamada al LLM")
            raise  # Re-raise para que el usuario vea el error
        
        # 7. Convertir respuesta de LiteLLM a nuestro formato
        logger.debug(f"[{request_id}] Construyendo respuesta...")
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
            logger.info(f"[{request_id}] 📊 Tokens: {response_dict['usage']['total_tokens']} total")
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] ✅ Respuesta generada exitosamente en {elapsed:.2f}s")
        logger.info(f"{'='*60}\n")
        return response_dict
        
    except HTTPException:
        # Re-raise HTTP exceptions sin alterar
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] ⏱️ Request falló en {elapsed:.2f}s")
        logger.info(f"{'='*60}\n")
        raise
        
    except ValueError as e:
        # Errores de validación
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] ❌ Validation error: {str(e)}")
        logger.error(f"[{request_id}] {traceback.format_exc()}")
        logger.info(f"[{request_id}] ⏱️ Request falló en {elapsed:.2f}s")
        logger.info(f"{'='*60}\n")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        # Errores generales
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] ❌ ERROR INESPERADO: {type(e).__name__}")
        logger.error(f"[{request_id}] Mensaje: {str(e)}")
        logger.error(f"[{request_id}] {traceback.format_exc()}")
        logger.info(f"[{request_id}] ⏱️ Request falló en {elapsed:.2f}s")
        logger.info(f"{'='*60}\n")
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
