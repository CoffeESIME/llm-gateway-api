"""
API de Embeddings Multimodales - Standalone
Uso: python standalone_embeddings_api.py
"""
import os
# --- FIX CRÍTICO PARA WINDOWS ---
# Descomentado para evitar que se congele la carga de modelos
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
# -------------------------------

import io
import torch
import librosa
import soundfile as sf
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image

# Librerías de Modelos
from sentence_transformers import SentenceTransformer
from transformers import SiglipProcessor, SiglipModel, ClapProcessor, ClapModel

# ==========================================
# CONFIGURACIÓN
# ==========================================
EMBEDDING_MODELS = {
    "text": {
        "model_id": "BAAI/bge-m3",
        "library": "sentence-transformers",
        "dimensions": 1024
    },
    "image": {
        "model_id": "google/siglip-so400m-patch14-384",
        "library": "transformers",
        "dimensions": 1152
    },
    "audio": {
        "model_id": "laion/clap-htsat-unfused",
        "library": "transformers",
        "dimensions": 512,
        "sample_rate": 48000
    }
}

MODELS = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Iniciando API Gateway en dispositivo: {DEVICE.upper()}")

# ==========================================
# LIFECYCLE
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ⚡ Optimizaciones GPU
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 1. TEXTO (BGE-M3)
    print(f"⏳ Cargando {EMBEDDING_MODELS['text']['model_id']} (Texto)...")
    # Usamos use_safetensors=True para evitar el error del .bin
    MODELS["text"] = SentenceTransformer(
        EMBEDDING_MODELS["text"]["model_id"], 
        device=DEVICE,
    )
    if DEVICE == "cuda": MODELS["text"].half()

    # 2. IMAGEN (SigLIP)
    print(f"⏳ Cargando {EMBEDDING_MODELS['image']['model_id']} (Imagen)...")
    MODELS["siglip_processor"] = SiglipProcessor.from_pretrained(EMBEDDING_MODELS["image"]["model_id"])
    MODELS["siglip_model"] = SiglipModel.from_pretrained(
        EMBEDDING_MODELS["image"]["model_id"],
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)

    # 3. AUDIO (CLAP)
    print(f"⏳ Cargando {EMBEDDING_MODELS['audio']['model_id']} (Audio)...")
    MODELS["clap_processor"] = ClapProcessor.from_pretrained(EMBEDDING_MODELS["audio"]["model_id"])
    try:
        # Intentamos cargar con safetensors primero
        MODELS["clap_model"] = ClapModel.from_pretrained(
            EMBEDDING_MODELS["audio"]["model_id"],
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)
    except Exception as e:
        print(f"⚠️ Aviso: CLAP no tiene safetensors, intentando carga standard... ({e})")
        # Si falla, intentamos carga normal (podría requerir PyTorch update si falla aquí)
        MODELS["clap_model"] = ClapModel.from_pretrained(
            EMBEDDING_MODELS["audio"]["model_id"],
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)

    print("✅ Todos los modelos cargados. API lista en http://localhost:8001")
    yield
    MODELS.clear()
app = FastAPI(
    title="Local Embeddings Gateway", 
    description="API de embeddings multimodales usando modelos open-source locales",
    version="1.0.0",
    lifespan=lifespan
)

# ==========================================
# SCHEMAS
# ==========================================

class TextRequest(BaseModel):
    text: str = Field(..., description="Texto a convertir en embedding")
    normalize: bool = Field(default=True, description="Normalizar vector")


class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., description="Lista de textos")
    normalize: bool = Field(default=True, description="Normalizar vectores")


class EmbeddingResponse(BaseModel):
    object: str = "embedding"
    model: str
    embedding: List[float]
    dimensions: int


class BatchEmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    data: List[dict]
    total: int


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "service": "Local Embeddings Gateway",
        "version": "1.0.0",
        "device": DEVICE,
        "models": {
            "text": EMBEDDING_MODELS["text"]["model_id"],
            "image": EMBEDDING_MODELS["image"]["model_id"],
            "audio": EMBEDDING_MODELS["audio"]["model_id"]
        },
        "endpoints": {
            "text": "/v1/embeddings/text",
            "text_batch": "/v1/embeddings/text/batch",
            "image": "/v1/embeddings/image",
            "audio": "/v1/embeddings/audio",
            "info": "/v1/embeddings/models"
        },
        "docs": "/docs"
    }


@app.get("/v1/embeddings/models")
async def get_models_info():
    """Retorna información sobre los modelos cargados"""
    return {
        "device": DEVICE,
        "models": EMBEDDING_MODELS
    }


@app.post("/v1/embeddings/text", response_model=EmbeddingResponse)
async def embed_text(request: TextRequest):
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
        # BGE-M3 en sentence-transformers es directo
        vector = MODELS["text"].encode(
            request.text, 
            normalize_embeddings=request.normalize
        )
        
        return EmbeddingResponse(
            model=EMBEDDING_MODELS["text"]["model_id"],
            embedding=vector.tolist(),
            dimensions=len(vector)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings/text/batch", response_model=BatchEmbeddingResponse)
async def embed_text_batch(request: BatchTextRequest):
    """
    Genera embeddings para múltiples textos en batch
    
    **Ejemplo:**
    ```json
    {
        "texts": ["Frase 1", "Frase 2", "Frase 3"],
        "normalize": true
    }
    ```
    """
    try:
        vectors = MODELS["text"].encode(
            request.texts,
            normalize_embeddings=request.normalize,
            show_progress_bar=False
        )
        
        data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": vector.tolist()
            }
            for i, vector in enumerate(vectors)
        ]
        
        return BatchEmbeddingResponse(
            model=EMBEDDING_MODELS["text"]["model_id"],
            data=data,
            total=len(data)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings/image", response_model=EmbeddingResponse)
async def embed_image(
    file: UploadFile = File(...),
    normalize: bool = True
):
    """
    Genera embedding (1152 dim) para imagen usando SigLIP
    
    **Soporta:** JPG, PNG, WEBP, etc.
    """
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Tipo inválido: {file.content_type}. Se esperaba imagen."
            )
        
        # Leer imagen desde memoria
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        processor = MODELS["siglip_processor"]
        model = MODELS["siglip_model"]

        # Procesar
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            
            if normalize:
                # Normalizar vector (importante para búsquedas por coseno)
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
        vector = outputs[0].cpu().tolist()
        
        return EmbeddingResponse(
            model=EMBEDDING_MODELS["image"]["model_id"],
            embedding=vector,
            dimensions=len(vector)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")


@app.post("/v1/embeddings/audio", response_model=EmbeddingResponse)
async def embed_audio(
    file: UploadFile = File(...),
    normalize: bool = True,
    max_duration: Optional[float] = None
):
    """
    Genera embedding (512 dim) para audio usando CLAP
    
    **Soporta:** WAV, MP3, FLAC, OGG, etc.
    
    **Parámetros:**
    - max_duration: Limita duración procesada (segundos)
    """
    try:
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail=f"Tipo inválido: {file.content_type}. Se esperaba audio."
            )
        
        TARGET_SR = EMBEDDING_MODELS["audio"]["sample_rate"]
        
        # Leer bytes y decodificar
        audio_bytes = await file.read()
        audio_data, samplerate = sf.read(io.BytesIO(audio_bytes))
        
        # Si es estéreo, pasar a mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Resamplear si no es 48k
        if samplerate != TARGET_SR:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=samplerate, 
                target_sr=TARGET_SR
            )
        
        # Limitar duración si se especifica
        if max_duration is not None:
            max_samples = int(max_duration * TARGET_SR)
            audio_data = audio_data[:max_samples]

        processor = MODELS["clap_processor"]
        model = MODELS["clap_model"]

        # Procesar
        inputs = processor(
            audios=audio_data, 
            sampling_rate=TARGET_SR, 
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.get_audio_features(**inputs)
            
            if normalize:
                # Normalizar
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

        vector = outputs[0].cpu().tolist()

        return EmbeddingResponse(
            model=EMBEDDING_MODELS["audio"]["model_id"],
            embedding=vector,
            dimensions=len(vector)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando audio: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Correr en puerto 8001 para no chocar con otros servicios
    uvicorn.run(app, host="0.0.0.0", port=8001)
