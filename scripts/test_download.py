import os
# Desactivar paralelismo para evitar bloqueos
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("1. Iniciando script de prueba...")

from sentence_transformers import SentenceTransformer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"2. Dispositivo detectado: {DEVICE}")

def test_load():
    print("3. Intentando descargar/cargar BAAI/bge-m3...")
    
    # Intentamos cargar forzando safetensors
    try:
        model = SentenceTransformer(
            "BAAI/bge-m3", 
            device=DEVICE,
        )
        print("4. ¡ÉXITO! El modelo se cargó correctamente.")
        
        # Prueba rápida
        embedding = model.encode("Hola mundo")
        print(f"5. Prueba de vector generada. Dimensión: {len(embedding)}")
        
    except Exception as e:
        print(f"❌ ERROR FATAL: {e}")

if __name__ == "__main__":
    # Esta protección es vital en Windows
    test_load()
