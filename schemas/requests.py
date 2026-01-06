"""
Esquemas de request para el API Gateway
Compatible con formato OpenAI Chat Completions con extensiones multimodales
"""
from pydantic import BaseModel, Field
from typing import List, Union, Literal, Optional, Dict, Any


# ==========================================
# CONTENIDO MULTIMODAL
# ==========================================

class TextContent(BaseModel):
    """Contenido de tipo texto"""
    type: Literal["text"] = "text"
    text: str


class FileReference(BaseModel):
    """Referencia a archivo adjunto por índice"""
    type: Literal["image", "audio", "document"]
    file_index: int = Field(..., description="Índice del archivo en la lista de archivos adjuntos")


# Union type para contenido mixto (texto o referencia a archivo)
ContentPart = Union[TextContent, FileReference]


class ChatMessage(BaseModel):
    """Mensaje individual en el chat"""
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentPart]]
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "role": "user",
                    "content": "Hola, ¿cómo estás?"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "¿Qué ves en esta imagen?"},
                        {"type": "image", "file_index": 0}
                    ]
                }
            ]
        }


# ==========================================
# REQUEST PRINCIPAL
# ==========================================

class ChatCompletionRequest(BaseModel):
    """
    Request para chat completions multimodal con archivos adjuntos
    
    **Formato:** multipart/form-data
    
    **Características:**
    - Soporta texto, imágenes, audio y video
    - Archivos < 5MB: Base64 data URI
    - Archivos >= 5MB: Google File API (requiere privacy_mode=flexible)
    - Routing automático entre modelos locales y cloud
    
    **Tasks disponibles:**
    - `chat`: Conversación general
    - `vision`: Análisis de imágenes/video
    - `ocr`: Extracción de texto de imágenes
    
    **Privacy Modes:**
    - `strict`: Solo modelos locales (Ollama) - máxima privacidad
    - `flexible`: Puede usar modelos cloud (Gemini) - máximo rendimiento
    """
    # Parámetros de routing
    task: Literal["chat", "vision", "ocr"] = Field(
        ...,
        description="Tipo de tarea a realizar (determina qué modelo usar)"
    )
    privacy_mode: Literal["strict", "flexible"] = Field(
        ...,
        description="Modo de privacidad - strict: solo local, flexible: permite cloud"
    )
    
    # Parámetros de chat
    messages: List[ChatMessage] = Field(
        ...,
        description="Lista de mensajes en la conversación. Puede incluir texto y referencias a archivos (file_index)"
    )
    
    # Parámetros opcionales
    model: Optional[str] = Field(
        None,
        description="Override manual del modelo (opcional, normalmente se auto-selecciona)"
    )
    temperature: Optional[float] = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Temperatura de generación (0.0 = determinista, 2.0 = muy creativo)"
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="Máximo número de tokens a generar"
    )
    stream: Optional[bool] = Field(
        False,
        description="Si se desea streaming de la respuesta (no soportado actualmente)"
    )
    top_p: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (alternativa a temperature)"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "task": "chat",
                    "privacy_mode": "strict",
                    "messages": [
                        {"role": "user", "content": "Resume este documento confidencial"}
                    ],
                    "temperature": 0.7
                },
                {
                    "task": "vision",
                    "privacy_mode": "flexible",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "¿Qué ves en esta imagen?"},
                                {"type": "image", "file_index": 0}
                            ]
                        }
                    ]
                },
                {
                    "task": "ocr",
                    "privacy_mode": "strict",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extrae el texto"},
                                {"type": "image", "file_index": 0}
                            ]
                        }
                    ],
                    "temperature": 0.0
                }
            ]
        }
