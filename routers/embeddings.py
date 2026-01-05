"""
Endpoints para generación de embeddings multimodales
Compatible con formato OpenAI Embeddings API
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from services.embedding_service import embedding_service


router = APIRouter(prefix="/v1/embeddings", tags=["Embeddings"])


# ==========================================
# SCHEMAS
# ==========================================

class TextEmbeddingRequest(BaseModel):
    """Request para embedding de texto"""
    text: str = Field(..., description="Texto a convertir en embedding")
    normalize: bool = Field(default=True, description="Normalizar vector (recomendado)")


class BatchTextEmbeddingRequest(BaseModel):
    """Request para embeddings de múltiples textos"""
    texts: List[str] = Field(..., description="Lista de textos a embedder")
    normalize: bool = Field(default=True, description="Normalizar vectores")


class EmbeddingResponse(BaseModel):
    """Respuesta compatible con formato OpenAI"""
    object: str = "embedding"
    model: str
    embedding: List[float]
    dimensions: int


class BatchEmbeddingResponse(BaseModel):
    """Respuesta para múltiples embeddings"""
    object: str = "list"
    model: str
    data: List[dict]
    total: int


# ==========================================
# ENDPOINTS - TEXTO
# ==========================================

@router.post("/text", response_model=EmbeddingResponse)
async def create_text_embedding(request: TextEmbeddingRequest):
    """
    Genera embedding denso (1024 dim) para texto usando BGE-M3
    
    **Ejemplo:**
    ```json
    {
        "text": "El perro corre por el parque",
        "normalize": true
    }
    ```
    """
    try:
        vector = embedding_service.embed_text(
            text=request.text,
            normalize=request.normalize
        )
        
        return EmbeddingResponse(
            model="BAAI/bge-m3",
            embedding=vector,
            dimensions=len(vector)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text/batch", response_model=BatchEmbeddingResponse)
async def create_text_embeddings_batch(request: BatchTextEmbeddingRequest):
    """
    Genera embeddings para múltiples textos en batch (más eficiente)
    
    **Ejemplo:**
    ```json
    {
        "texts": [
            "Primera frase",
            "Segunda frase",
            "Tercera frase"
        ],
        "normalize": true
    }
    ```
    """
    try:
        vectors = embedding_service.embed_texts_batch(
            texts=request.texts,
            normalize=request.normalize
        )
        
        data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": vector
            }
            for i, vector in enumerate(vectors)
        ]
        
        return BatchEmbeddingResponse(
            model="BAAI/bge-m3",
            data=data,
            total=len(vectors)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINTS - IMAGEN
# ==========================================

@router.post("/image", response_model=EmbeddingResponse)
async def create_image_embedding(
    file: UploadFile = File(..., description="Archivo de imagen (JPG, PNG, etc.)"),
    normalize: bool = True
):
    """
    Genera embedding (1152 dim) para imagen usando SigLIP
    
    **Soporta:** JPG, PNG, WEBP, etc.
    """
    try:
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail=f"Tipo de archivo inválido: {file.content_type}. Se esperaba una imagen."
            )
        
        image_bytes = await file.read()
        
        vector = embedding_service.embed_image(
            image_bytes=image_bytes,
            normalize=normalize
        )
        
        return EmbeddingResponse(
            model="google/siglip-so400m-patch14-384",
            embedding=vector,
            dimensions=len(vector)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")


# ==========================================
# ENDPOINTS - AUDIO
# ==========================================

@router.post("/audio", response_model=EmbeddingResponse)
async def create_audio_embedding(
    file: UploadFile = File(..., description="Archivo de audio (WAV, MP3, FLAC, etc.)"),
    normalize: bool = True,
    max_duration: Optional[float] = None
):
    """
    Genera embedding (512 dim) para audio usando CLAP
    
    **Soporta:** WAV, MP3, FLAC, OGG, etc.
    
    **Parámetros:**
    - `max_duration`: Limita la duración del audio procesado (en segundos)
    """
    try:
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo inválido: {file.content_type}. Se esperaba audio."
            )
        
        audio_bytes = await file.read()
        
        vector = embedding_service.embed_audio(
            audio_bytes=audio_bytes,
            normalize=normalize,
            max_duration=max_duration
        )
        
        return EmbeddingResponse(
            model="laion/clap-htsat-unfused",
            embedding=vector,
            dimensions=len(vector)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando audio: {str(e)}")


# ==========================================
# ENDPOINT - INFO
# ==========================================

@router.get("/models")
async def get_models_info():
    """
    Retorna información sobre los modelos de embedding cargados
    
    Incluye:
    - IDs de modelos
    - Dimensiones de vectores
    - Librerías utilizadas
    - Dispositivo de ejecución (GPU/CPU)
    """
    try:
        return embedding_service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
