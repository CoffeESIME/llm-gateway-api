"""
Servicio de Embeddings Multimodales
Soporta texto, imagen y audio usando modelos locales
"""
import io
import torch
import librosa
import soundfile as sf
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
import os
os.environ["HF_HUB_OFFLINE"] = "1"
# Librerías de Modelos
from sentence_transformers import SentenceTransformer
from transformers import SiglipProcessor, SiglipModel, ClapProcessor, ClapModel

from config import settings


class EmbeddingService:
    """
    Servicio singleton para gestionar modelos de embedding multimodales.
    Carga los modelos una sola vez y los mantiene en memoria (VRAM si está disponible).
    """
    
    _instance = None
    _models: Dict[str, Any] = {}
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inicializa el servicio si no ha sido inicializado previamente"""
        if not self._initialized:
            print(f"🚀 Inicializando EmbeddingService en dispositivo: {self._device.upper()}")
            self._load_models()
            self._initialized = True
    
    def _load_models(self):
        """Carga todos los modelos de embedding en memoria con optimizaciones para GPU"""
        config = settings.EMBEDDING_MODELS
        
        # ⚡ Optimizaciones para GPU (especialmente RTX 4090)
        if self._device == "cuda":
            print("🚀 GPU detectada - Activando optimizaciones Turbo Mode:")
            
            # Habilitar cuDNN benchmark para operaciones más rápidas
            torch.backends.cudnn.benchmark = True
            print("   ✓ cuDNN benchmark habilitado")
            
            # Configurar para usar TF32 en Ampere/Ada (3090, 4090, etc.)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("   ✓ TF32 habilitado (Ampere/Ada Lovelace)")
        
        # 1. Cargar modelo de Texto (BGE-M3)
        print(f"⏳ Cargando {config['text']['model_id']} (Texto)...")
        self._models["text"] = SentenceTransformer(
            config["text"]["model_id"], 
            device=self._device
        )
        
        # ✨ OPTIMIZACIÓN: Convertir a FP16 para RTX 4090
        if self._device == "cuda":
            self._models["text"].half()
            print(f"   ✓ Modelo de texto convertido a FP16 (menor VRAM, mayor velocidad)")
        
        # 2. Cargar modelo de Imagen (SigLIP)
        print(f"⏳ Cargando {config['image']['model_id']} (Imagen)...")
        self._models["siglip_processor"] = SiglipProcessor.from_pretrained(
            config["image"]["model_id"]
        )
        
        # ✨ OPTIMIZACIÓN: Cargar directamente en torch.float16
        self._models["siglip_model"] = SiglipModel.from_pretrained(
            config["image"]["model_id"],
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
        ).to(self._device)
        
        if self._device == "cuda":
            print(f"   ✓ Modelo de imagen cargado en FP16")
        
        # 3. Cargar modelo de Audio (CLAP)
        print(f"⏳ Cargando {config['audio']['model_id']} (Audio)...")
        self._models["clap_processor"] = ClapProcessor.from_pretrained(
            config["audio"]["model_id"]
        )
        
        # ✨ OPTIMIZACIÓN: Cargar directamente en torch.float16
        self._models["clap_model"] = ClapModel.from_pretrained(
            config["audio"]["model_id"],
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
        ).to(self._device)
        
        if self._device == "cuda":
            print(f"   ✓ Modelo de audio cargado en FP16")
        
        print("✅ Todos los modelos de embedding cargados exitosamente.")
        if self._device == "cuda":
            print("🔥 Modo Turbo FP16 activado - Máximo rendimiento en RTX 4090")
    
    @property
    def device(self) -> str:
        """Retorna el dispositivo actual (cuda o cpu)"""
        return self._device
    
    def embed_text(self, text: str, normalize: bool = True) -> List[float]:
        """
        Genera embedding para texto usando BGE-M3
        
        Args:
            text: Texto a embedder
            normalize: Si True, normaliza el vector (recomendado para búsqueda por similitud)
            
        Returns:
            Vector de 1024 dimensiones como lista de floats
        """
        try:
            vector = self._models["text"].encode(
                text, 
                normalize_embeddings=normalize
            )
            return vector.tolist()
        except Exception as e:
            raise RuntimeError(f"Error generando embedding de texto: {str(e)}")
    
    def embed_texts_batch(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos en batch
        
        Args:
            texts: Lista de textos a embedder
            normalize: Si True, normaliza los vectores
            
        Returns:
            Lista de vectores de 1024 dimensiones
        """
        try:
            vectors = self._models["text"].encode(
                texts, 
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            return vectors.tolist()
        except Exception as e:
            raise RuntimeError(f"Error generando embeddings de texto en batch: {str(e)}")
    
    def embed_image(self, image_bytes: bytes, normalize: bool = True) -> List[float]:
        """
        Genera embedding para imagen usando SigLIP
        
        Args:
            image_bytes: Bytes de la imagen
            normalize: Si True, normaliza el vector
            
        Returns:
            Vector de 1152 dimensiones como lista de floats
        """
        try:
            # Cargar imagen desde bytes
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Procesar imagen
            processor = self._models["siglip_processor"]
            model = self._models["siglip_model"]
            
            inputs = processor(images=image, return_tensors="pt").to(self._device)
            
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                
                if normalize:
                    # Normalizar vector L2 (importante para búsquedas por coseno)
                    outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            vector = outputs[0].cpu().tolist()
            return vector
            
        except Exception as e:
            raise RuntimeError(f"Error generando embedding de imagen: {str(e)}")
    
    def embed_audio(
        self, 
        audio_bytes: bytes, 
        normalize: bool = True,
        max_duration: Optional[float] = None
    ) -> List[float]:
        """
        Genera embedding para audio usando CLAP
        
        Args:
            audio_bytes: Bytes del archivo de audio
            normalize: Si True, normaliza el vector
            max_duration: Duración máxima en segundos (None = sin límite)
            
        Returns:
            Vector de 512 dimensiones como lista de floats
        """
        try:
            TARGET_SR = settings.EMBEDDING_MODELS["audio"]["sample_rate"]
            
            # Leer y decodificar audio
            audio_data, samplerate = sf.read(io.BytesIO(audio_bytes))
            
            # Convertir estéreo a mono si es necesario
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resamplear si no es la frecuencia esperada
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
            
            # Procesar audio
            processor = self._models["clap_processor"]
            model = self._models["clap_model"]
            
            inputs = processor(
                audios=audio_data, 
                sampling_rate=TARGET_SR, 
                return_tensors="pt"
            ).to(self._device)
            
            with torch.no_grad():
                outputs = model.get_audio_features(**inputs)
                
                if normalize:
                    # Normalizar vector L2
                    outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            vector = outputs[0].cpu().tolist()
            return vector
            
        except Exception as e:
            raise RuntimeError(f"Error generando embedding de audio: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información sobre los modelos cargados
        
        Returns:
            Diccionario con información de modelos y configuración
        """
        return {
            "device": self._device,
            "models": {
                "text": {
                    "model_id": settings.EMBEDDING_MODELS["text"]["model_id"],
                    "dimensions": settings.EMBEDDING_MODELS["text"]["dimensions"],
                    "library": settings.EMBEDDING_MODELS["text"]["library"]
                },
                "image": {
                    "model_id": settings.EMBEDDING_MODELS["image"]["model_id"],
                    "dimensions": settings.EMBEDDING_MODELS["image"]["dimensions"],
                    "library": settings.EMBEDDING_MODELS["image"]["library"]
                },
                "audio": {
                    "model_id": settings.EMBEDDING_MODELS["audio"]["model_id"],
                    "dimensions": settings.EMBEDDING_MODELS["audio"]["dimensions"],
                    "library": settings.EMBEDDING_MODELS["audio"]["library"],
                    "sample_rate": settings.EMBEDDING_MODELS["audio"]["sample_rate"]
                }
            }
        }


# Instancia global del servicio (singleton)
embedding_service = EmbeddingService()
