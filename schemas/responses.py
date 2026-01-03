"""
Esquemas de response compatibles con formato OpenAI
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Message(BaseModel):
    """Mensaje en la respuesta"""
    role: str
    content: str


class Choice(BaseModel):
    """Choice individual en la respuesta"""
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    """Información de uso de tokens"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Respuesta de chat completion compatible con formato OpenAI"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "ollama/dolphin-mistral-nemo:latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hola, ¿en qué puedo ayudarte?"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
        }
