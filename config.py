"""
Configuración centralizada para el LLM Gateway API
"""
from pydantic_settings import BaseSettings
from typing import Dict, Any


class Settings(BaseSettings):
    """Configuración de la aplicación usando Pydantic Settings"""
    
    # API Keys
    gemini_api_key: str = ""
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    
    # Logging
    log_level: str = "INFO"
    
    # File Processing Configuration
    LARGE_FILE_THRESHOLD: int = 5 * 1024 * 1024  # 5MB - archivos mayores usan Google File API
    
    # Model Router Configuration
    # Mapeo de tareas y modos de privacidad a modelos específicos
    MODEL_ROUTER: Dict[str, Dict[str, str]] = {
        "chat": {
            "strict": "ollama/gpt-oss:20b",
            "flexible": "gemini/gemini-2.5-flash"
        },
        "vision": {
            "strict": "ollama/qwen3-vl:8b",
            "flexible": "gemini/gemini-2.5-flash"  # Corregido: agregado prefijo gemini/
        },
        "ocr": {
            "strict": "ollama/qwen3-vl:8b",
            "flexible": "gemini/gemini-2.5-flash"
        },
    }
    
    # Configuración de Modelos de Embedding Multimodales
    EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
        "text": {
            "model_id": "BAAI/bge-m3",
            "library": "sentence-transformers",  # Genera vectores de 1024 dimensiones
            "dimensions": 1024
        },
        "image": {
            "model_id": "google/siglip-so400m-patch14-384",
            "library": "transformers",  # Genera vectores de 1152 dimensiones
            "dimensions": 1152
        },
        "audio": {
            "model_id": "laion/clap-htsat-unfused",
            "library": "transformers",  # Genera vectores de 512 dimensiones
            "dimensions": 512,
            "sample_rate": 48000  # CLAP requiere audio a 48kHz
        }
    }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instancia global de configuración
settings = Settings()
