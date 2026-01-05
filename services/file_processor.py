"""
Procesador de archivos multimedia para chat completions
Convierte archivos a formatos compatibles con LLMs
Soporta Google File API para archivos grandes
"""
import base64
import io
from typing import Dict, Any, List, Optional
from fastapi import UploadFile
from PIL import Image

from config import settings
from services.google_file_api import google_file_api, GoogleFileAPIError


# Límites de archivos
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB (límite de Google File API)
LARGE_FILE_THRESHOLD = settings.LARGE_FILE_THRESHOLD  # 5MB

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/mp3", "audio/mpeg", "audio/flac", "audio/ogg", "audio/m4a"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/mpeg", "video/webm", "video/mov"}
ALLOWED_DOCUMENT_TYPES = {"application/pdf", "text/plain", "image/jpeg", "image/png"}


async def process_uploaded_file(file: UploadFile, file_type: str) -> Dict[str, Any]:
    """
    Procesa un archivo subido y retorna su información
    
    Args:
        file: Archivo subido via FastAPI
        file_type: Tipo esperado ("image", "audio", "video", "document")
    
    Returns:
        Dict con metadata del archivo procesado
    
    Raises:
        ValueError: Si el archivo no es válido
    """
    # Leer contenido
    content = await file.read()
    file_size = len(content)
    
    # Validar tamaño
    if file_size > MAX_FILE_SIZE:
        raise ValueError(
            f"Archivo demasiado grande: {file_size / 1024 / 1024:.1f}MB. "
            f"Máximo permitido: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        )
    
    # Validar tipo MIME
    content_type = file.content_type or "application/octet-stream"
    
    if file_type == "image" and content_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError(
            f"Tipo de imagen no soportado: {content_type}. "
            f"Soportados: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )
    elif file_type == "audio" and content_type not in ALLOWED_AUDIO_TYPES:
        raise ValueError(
            f"Tipo de audio no soportado: {content_type}. "
            f"Soportados: {', '.join(ALLOWED_AUDIO_TYPES)}"
        )
    elif file_type == "video" and content_type not in ALLOWED_VIDEO_TYPES:
        raise ValueError(
            f"Tipo de video no soportado: {content_type}. "
            f"Soportados: {', '.join(ALLOWED_VIDEO_TYPES)}"
        )
    
    # Retornar metadata
    return {
        "filename": file.filename,
        "content_type": content_type,
        "size": file_size,
        "content": content,
        "file_type": file_type
    }


def file_to_base64(file_content: bytes, content_type: str) -> str:
    """
    Convierte contenido de archivo a data URI base64
    
    Args:
        file_content: Bytes del archivo
        content_type: MIME type del archivo
    
    Returns:
        String data URI (data:image/jpeg;base64,...)
    """
    base64_data = base64.b64encode(file_content).decode('utf-8')
    return f"data:{content_type};base64,{base64_data}"


def prepare_multimodal_content(
    messages: List[Dict], 
    files_metadata: List[Dict[str, Any]],
    privacy_mode: str = "strict"
) -> List[Dict]:
    """
    Prepara contenido multimodal para envío al LLM
    
    Estrategia:
    - Archivos pequeños (< 5MB): Base64 data URI
    - Archivos grandes (>= 5MB):
        * privacy_mode=flexible: Google File API (sube a Google, usa URI)
        * privacy_mode=strict: NotImplementedError (TODO: chunking local)
    
    Args:
        messages: Lista de mensajes con posibles file_index
        files_metadata: Metadata de archivos procesados
        privacy_mode: "strict" (local) o "flexible" (cloud)
    
    Returns:
        Mensajes preparados para el LLM
    
    Raises:
        NotImplementedError: Si archivo grande con privacy_mode=strict
        ValueError: Si file_index inválido
    """
    prepared_messages = []
    
    for msg in messages:
        prepared_msg = {"role": msg["role"]}
        
        # Si es string simple, pasar directo
        if isinstance(msg["content"], str):
            prepared_msg["content"] = msg["content"]
            prepared_messages.append(prepared_msg)
            continue
        
        # Si es lista de contenido, procesar cada parte
        prepared_content = []
        for part in msg["content"]:
            if part.get("type") == "text":
                # Texto plano
                prepared_content.append(part)
            
            elif part.get("type") in ["image", "audio", "video", "document"]:
                # Archivo referenciado
                file_index = part.get("file_index")
                
                if file_index is None or file_index >= len(files_metadata):
                    raise ValueError(
                        f"file_index {file_index} inválido. "
                        f"Se esperaban {len(files_metadata)} archivos."
                    )
                
                file_meta = files_metadata[file_index]
                file_size = file_meta["size"]
                content_type = file_meta["content_type"]
                file_type = part["type"]
                
                # DECISIÓN: ¿Archivo grande o pequeño?
                if file_size >= LARGE_FILE_THRESHOLD:
                    # ========================================
                    # CASO 1: ARCHIVO GRANDE (>= 5MB)
                    # ========================================
                    
                    if privacy_mode == "flexible":
                        # A) Modo Flexible (Nube de Google) -> Usar File API
                        try:
                            file_uri = google_file_api.upload_file(
                                file_bytes=file_meta["content"],
                                filename=file_meta["filename"],
                                mime_type=content_type
                            )
                            
                            # Formato para Gemini con File API
                            prepared_content.append({
                                "type": "file_data",
                                "file_data": {
                                    "file_uri": file_uri,
                                    "mime_type": content_type
                                }
                            })
                            
                        except GoogleFileAPIError as e:
                            raise ValueError(
                                f"Error subiendo archivo a Google File API: {str(e)}"
                            )
                    
                    elif privacy_mode == "strict":
                        # B) Modo Estricto (Local/Privado) -> TODO: Chunking local
                        raise NotImplementedError(
                            f"Procesamiento local de archivos grandes ({file_size / 1024 / 1024:.1f}MB) "
                            f"está en desarrollo.\n\n"
                            f"TODO Roadmap:\n"
                            f"- Video: Extraer frames cada N segundos con ffmpeg -> enviar como lista de imágenes\n"
                            f"- Audio: Dividir en chunks de 10min con ffmpeg -> transcribir secuencialmente con Whisper\n\n"
                            f"Por ahora, usa privacy_mode='flexible' para archivos grandes."
                        )
                
                else:
                    # ========================================
                    # CASO 2: ARCHIVO PEQUEÑO (< 5MB)
                    # ========================================
                    
                    # Para archivos pequeños, usar base64 (funciona con cualquier privacy_mode)
                    if file_type in ["image", "document"]:
                        # Imágenes y documentos: formato image_url con data URI
                        data_uri = file_to_base64(file_meta["content"], content_type)
                        prepared_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri
                            }
                        })
                    
                    elif file_type == "audio":
                        # Audio pequeño: también como data URI si el modelo lo soporta
                        # Alternativamente, podrías transcribirlo localmente con Whisper
                        data_uri = file_to_base64(file_meta["content"], content_type)
                        prepared_content.append({
                            "type": "audio_url",
                            "audio_url": {
                                "url": data_uri
                            }
                        })
                    
                    elif file_type == "video":
                        # Video pequeño: placeholder por ahora
                        # TODO: Extraer frame representativo o usar File API
                        prepared_content.append({
                            "type": "text",
                            "text": f"[Video file: {file_meta['filename']} - {file_size / 1024:.1f}KB]"
                        })
        
        prepared_msg["content"] = prepared_content
        prepared_messages.append(prepared_msg)
    
    return prepared_messages
