"""
Script de prueba para los endpoints de embeddings multimodales
Prueba los tres tipos de embeddings: texto, imagen y audio
"""
import requests
import json
from pathlib import Path

# Configuración
BASE_URL = "http://localhost:8765"
API_BASE = f"{BASE_URL}/v1/embeddings"


def test_models_info():
    """Prueba el endpoint de información de modelos"""
    print("\n" + "="*60)
    print("🔍 Probando endpoint de información de modelos...")
    print("="*60)
    
    response = requests.get(f"{API_BASE}/models")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Modelos cargados en dispositivo: {data['device'].upper()}")
        print("\n📊 Configuración de modelos:")
        for model_type, config in data['models'].items():
            print(f"\n  {model_type.upper()}:")
            print(f"    - Model ID: {config['model_id']}")
            print(f"    - Dimensiones: {config['dimensions']}")
            print(f"    - Librería: {config['library']}")
            if 'sample_rate' in config:
                print(f"    - Sample Rate: {config['sample_rate']} Hz")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def test_text_embedding():
    """Prueba el embedding de texto"""
    print("\n" + "="*60)
    print("📝 Probando embedding de texto...")
    print("="*60)
    
    payload = {
        "text": "El perro corre felizmente por el parque verde",
        "normalize": True
    }
    
    response = requests.post(f"{API_BASE}/text", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Embedding generado exitosamente")
        print(f"   Modelo: {data['model']}")
        print(f"   Dimensiones: {data['dimensions']}")
        print(f"   Vector (primeros 5 valores): {data['embedding'][:5]}")
        print(f"   Vector (últimos 5 valores): {data['embedding'][-5:]}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def test_batch_text_embedding():
    """Prueba el embedding de texto en batch"""
    print("\n" + "="*60)
    print("📝 Probando embedding de texto en BATCH...")
    print("="*60)
    
    payload = {
        "texts": [
            "El gato negro duerme en la cama",
            "Un perro grande corre por la playa",
            "Las aves vuelan alto en el cielo azul"
        ],
        "normalize": True
    }
    
    response = requests.post(f"{API_BASE}/text/batch", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Batch de embeddings generado exitosamente")
        print(f"   Modelo: {data['model']}")
        print(f"   Total de vectores: {data['total']}")
        for item in data['data']:
            print(f"   - Índice {item['index']}: {len(item['embedding'])} dimensiones")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def test_image_embedding(image_path: str = None):
    """Prueba el embedding de imagen"""
    print("\n" + "="*60)
    print("🖼️  Probando embedding de imagen...")
    print("="*60)
    
    if image_path is None or not Path(image_path).exists():
        print("⚠️  No se proporcionó imagen de prueba válida")
        print("   Para probar con una imagen real, ejecuta:")
        print("   test_image_embedding('ruta/a/tu/imagen.jpg')")
        return
    
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        response = requests.post(f"{API_BASE}/image", files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Embedding de imagen generado exitosamente")
        print(f"   Modelo: {data['model']}")
        print(f"   Dimensiones: {data['dimensions']}")
        print(f"   Vector (primeros 5 valores): {data['embedding'][:5]}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def test_audio_embedding(audio_path: str = None):
    """Prueba el embedding de audio"""
    print("\n" + "="*60)
    print("🎵 Probando embedding de audio...")
    print("="*60)
    
    if audio_path is None or not Path(audio_path).exists():
        print("⚠️  No se proporcionó archivo de audio de prueba válido")
        print("   Para probar con un audio real, ejecuta:")
        print("   test_audio_embedding('ruta/a/tu/audio.wav')")
        return
    
    with open(audio_path, 'rb') as f:
        files = {'file': (Path(audio_path).name, f, 'audio/wav')}
        response = requests.post(f"{API_BASE}/audio", files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Embedding de audio generado exitosamente")
        print(f"   Modelo: {data['model']}")
        print(f"   Dimensiones: {data['dimensions']}")
        print(f"   Vector (primeros 5 valores): {data['embedding'][:5]}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def main():
    """Ejecuta todas las pruebas"""
    print("\n" + "="*60)
    print("🚀 PRUEBAS DE EMBEDDINGS MULTIMODALES")
    print("="*60)
    print(f"API Base URL: {API_BASE}")
    
    try:
        # Probar conexión
        health_response = requests.get(f"{BASE_URL}/health")
        if health_response.status_code != 200:
            print("❌ Error: La API no está respondiendo")
            print("   Asegúrate de que el servidor esté corriendo:")
            print("   python main.py")
            return
        
        # Ejecutar pruebas
        test_models_info()
        test_text_embedding()
        test_batch_text_embedding()
        test_image_embedding()  # Fallará si no hay imagen
        test_audio_embedding()  # Fallará si no hay audio
        
        print("\n" + "="*60)
        print("✅ PRUEBAS COMPLETADAS")
        print("="*60)
        print("\n💡 Tips:")
        print("   - Para probar imágenes: test_image_embedding('imagen.jpg')")
        print("   - Para probar audio: test_audio_embedding('audio.wav')")
        print("   - Para benchmarks: python test_embeddings.py --benchmark")
        print("   - Documentación interactiva: http://localhost:8765/docs")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: No se pudo conectar a la API")
        print("   Asegúrate de que el servidor esté corriendo:")
        print("   python main.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


# ==========================================
# BENCHMARKS DE RENDIMIENTO
# ==========================================

def benchmark_text_single():
    """Benchmark de embeddings de texto individuales"""
    print("\n" + "="*60)
    print("📊 BENCHMARK: Embedding de Texto Individual")
    print("="*60)
    
    import time
    
    text = "El perro corre felizmente por el parque verde mientras el sol brilla"
    iterations = 100
    
    times = []
    for i in range(iterations):
        start = time.time()
        response = requests.post(
            f"{API_BASE}/text",
            json={"text": text, "normalize": True}
        )
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        
        if i == 0:
            print(f"   Primera inferencia: {elapsed:.2f}ms (incluye overhead)")
    
    # Excluir primera iteración (warmup)
    times = times[1:]
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    throughput = 1000 / avg_time  # embeddings por segundo
    
    print(f"\n   Iteraciones: {iterations-1}")
    print(f"   Tiempo promedio: {avg_time:.2f}ms")
    print(f"   Tiempo mínimo: {min_time:.2f}ms")
    print(f"   Tiempo máximo: {max_time:.2f}ms")
    print(f"   Throughput: {throughput:.1f} embeddings/segundo")
    print(f"   🚀 Embeddings por minuto: ~{throughput*60:.0f}")


def benchmark_text_batch():
    """Benchmark de batch processing"""
    print("\n" + "="*60)
    print("📊 BENCHMARK: Batch Processing de Textos")
    print("="*60)
    
    import time
    
    batch_sizes = [10, 50, 100, 200]
    
    for batch_size in batch_sizes:
        texts = [f"Este es un texto de prueba número {i}" for i in range(batch_size)]
        
        start = time.time()
        response = requests.post(
            f"{API_BASE}/text/batch",
            json={"texts": texts, "normalize": True}
        )
        elapsed = (time.time() - start) * 1000  # ms
        
        if response.status_code == 200:
            throughput = batch_size / (elapsed / 1000)  # textos por segundo
            time_per_embedding = elapsed / batch_size
            
            print(f"\n   Batch Size: {batch_size}")
            print(f"   Tiempo total: {elapsed:.2f}ms")
            print(f"   Tiempo por embedding: {time_per_embedding:.2f}ms")
            print(f"   Throughput: {throughput:.1f} embeddings/segundo")
        else:
            print(f"\n   ❌ Error con batch size {batch_size}: {response.status_code}")


def benchmark_comparison():
    """Compara rendimiento entre modalidades"""
    print("\n" + "="*60)
    print("📊 BENCHMARK: Comparación de Modalidades")
    print("="*60)
    
    import time
    
    results = {}
    
    # Texto
    start = time.time()
    response = requests.post(
        f"{API_BASE}/text",
        json={"text": "Texto de prueba", "normalize": True}
    )
    if response.status_code == 200:
        results["Texto (1024 dim)"] = (time.time() - start) * 1000
    
    # Nota: Imagen y Audio requieren archivos
    print("\n   Modalidad          | Tiempo (ms) | Dimensiones")
    print("   " + "-"*50)
    
    for modality, time_ms in results.items():
        print(f"   {modality:<18} | {time_ms:>10.2f} |")
    
    print("\n   ℹ️  Para benchmark completo con imágenes/audio:")
    print("      Agrega archivos de prueba y ejecuta de nuevo")


def run_all_benchmarks():
    """Ejecuta todos los benchmarks"""
    print("\n" + "="*60)
    print("🔥 BENCHMARKS DE RENDIMIENTO - RTX 4090")
    print("="*60)
    
    try:
        # Verificar conexión
        requests.get(f"{BASE_URL}/health")
        
        # Ejecutar benchmarks
        benchmark_text_single()
        benchmark_text_batch()
        benchmark_comparison()
        
        print("\n" + "="*60)
        print("✅ BENCHMARKS COMPLETADOS")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: No se pudo conectar a la API")
        print("   Inicia el servidor primero: python main.py")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        run_all_benchmarks()
    else:
        main()
