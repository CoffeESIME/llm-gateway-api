"""
Procesador de archivos multimedia para chat completions
Convierte archivos a formatos compatibles con LLMs
"""
import base64
import io
from typing import Dict, Any, List, Optional
from fastapi import UploadFile
from PIL import Image

from services.embedding_service import embedding_service


# Límites de archivos
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/mp3", "audio/mpeg", "audio/flac", "audio/ogg"}
ALLOWED_DOCUMENT_TYPES = {"application/pdf", "text/plain", "image/jpeg", "image/png"}


async def process_uploaded_file(file: UploadFile, file_type: str) -> Dict[str, Any]:
    """
    Procesa un archivo subido y retorna su información
    
    Args:
        file: Archivo subido via FastAPI
        file_type: Tipo esperado ("image", "audio", "document")
    
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
    
    # Retornar metadata
    return {
        "filename": file.filename,
        "content_type": content_type,
        "size": file_size,
        "content": content
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
    files_metadata: List[Dict[str, Any]]
) -> List[Dict]:
    """
    Prepara contenido multimodal para envío al LLM
    Reemplaza file_index con URLs data base64
    
    Args:
        messages: Lista de mensajes con posibles file_index
        files_metadata: Metadata de archivos procesados
    
    Returns:
        Mensajes preparados para el LLM
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
            
            elif part.get("type") in ["image", "audio", "document"]:
                # Archivo referenciado
                file_index = part.get("file_index")
                
                if file_index is None or file_index >= len(files_metadata):
                    raise ValueError(
                        f"file_index {file_index} inválido. "
                        f"Se esperaban {len(files_metadata)} archivos."
                    )
                
                file_meta = files_metadata[file_index]
                
                # Convertir a formato compatible con LLM
                if part["type"] == "image":
                    # Para imágenes, usar formato image_url con data URI
                    data_uri = file_to_base64(file_meta["content"], file_meta["content_type"])
                    prepared_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    })
                
                elif part["type"] == "audio":
                    # Para audio, podrías generar embedding o transcripción
                    # Por ahora, incluir como texto descriptivo
                    prepared_content.append({
                        "type": "text",
                        "text": f"[Audio file: {file_meta['filename']}]"
                    })
                
                else:  # document
                    # Para documentos, similar a audio
                    prepared_content.append({
                        "type": "text",
                        "text": f"[Document file: {file_meta['filename']}]"
                    })
        
        prepared_msg["content"] = prepared_content
        prepared_messages.append(prepared_msg)
    
    return prepared_messages
