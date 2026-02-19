"""
Script de prueba para validar el funcionamiento de embeddings.

Prueba 1: Calidad semántica del modelo BGE-M3
  - Compara texto crudo vs texto enriquecido con conceptos (Graph RAG)
  - Verifica que el enriquecimiento mejore la similitud semántica

Prueba 2: Funcionamiento via API
  - Llama a los endpoints REST del gateway para texto, batch e info

Uso local (directo):
  python scripts/test_embedding_quality.py

Uso via API (requiere servidor corriendo):
  python scripts/test_embedding_quality.py --api
"""
import argparse
import sys
import os

# Agregar el directorio raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_local_embedding_quality():
    """
    Prueba directa: carga el modelo via EmbeddingService del proyecto
    y valida calidad semántica con el caso de la "Cabra" de Terray.
    """
    import torch
    from services.embedding_service import embedding_service

    print("\n" + "=" * 60)
    print("🧪 PRUEBA 1: Calidad Semántica (BGE-M3 via EmbeddingService)")
    print("=" * 60)

    # Info del servicio
    info = embedding_service.get_model_info()
    print(f"📌 Dispositivo: {info['device'].upper()}")
    print(f"📌 Modelo texto: {info['models']['text']['model_id']}")
    print(f"📌 Dimensiones: {info['models']['text']['dimensions']}")

    # ─── Caso de prueba: Texto crudo vs enriquecido ───
    text_raw = (
        "Yo no era un alpinista intelectual, sino un animal fogoso "
        "que saltaba de cima en cima como una cabra."
    )

    text_enriched = (
        "Yo no era un alpinista intelectual, sino un animal fogoso "
        "que saltaba de cima en cima como una cabra. "
        "Conceptos Clave: Agilidad, Ímpetu, Libertad, Pasión, Instinto, Naturaleza."
    )

    # Query difícil: "Libertad" NO aparece en el texto crudo
    query = "La sensación de libertad física y destreza"

    print(f"\n🔎 QUERY: '{query}'")
    print(f"📄 TEXTO CRUDO: '{text_raw[:80]}...'")
    print(f"✨ TEXTO ENRIQUECIDO: '{text_enriched[:80]}...'")

    # ─── Generar embeddings ───
    print("\n🔄 Vectorizando con EmbeddingService...")
    vec_query = embedding_service.embed_text(query)
    vec_raw = embedding_service.embed_text(text_raw)
    vec_enriched = embedding_service.embed_text(text_enriched)

    # Verificar dimensiones
    assert len(vec_query) == info['models']['text']['dimensions'], \
        f"Dimensiones incorrectas: {len(vec_query)} vs {info['models']['text']['dimensions']}"

    # ─── Calcular similitud coseno ───
    t_query = torch.tensor(vec_query)
    t_raw = torch.tensor(vec_raw)
    t_enriched = torch.tensor(vec_enriched)

    score_raw = torch.nn.functional.cosine_similarity(t_query.unsqueeze(0), t_raw.unsqueeze(0)).item()
    score_enriched = torch.nn.functional.cosine_similarity(t_query.unsqueeze(0), t_enriched.unsqueeze(0)).item()

    # ─── Resultados ───
    print("\n" + "-" * 40)
    print(f"📄 TEXTO CRUDO (Score):      {score_raw:.4f}")
    print(f"✨ TEXTO ENRIQUECIDO (Score): {score_enriched:.4f}")
    print("-" * 40)

    diff = score_enriched - score_raw
    if diff > 0.05:
        print(f"✅ ÉXITO: El enriquecimiento mejoró la búsqueda en {(diff*100):.2f}%")
    elif diff > 0:
        print(f"⚠️ MEJORA LEVE: +{(diff*100):.2f}%")
    else:
        print("❌ FALLO: El enriquecimiento no mejoró la búsqueda")

    return score_raw, score_enriched


def test_batch_embeddings():
    """
    Prueba de batch embeddings y verificación de consistencia.
    """
    import torch
    from services.embedding_service import embedding_service

    print("\n" + "=" * 60)
    print("🧪 PRUEBA 2: Batch Embeddings")
    print("=" * 60)

    texts = [
        "El gato duerme en el sofá",
        "El perro corre por el parque",
        "Python es un lenguaje de programación",
        "La temperatura del reactor es de 350 grados",
    ]

    print(f"📝 Procesando {len(texts)} textos en batch...")
    vectors = embedding_service.embed_texts_batch(texts)

    assert len(vectors) == len(texts), f"Se esperaban {len(texts)} vectores, se obtuvieron {len(vectors)}"
    print(f"✅ Vectores generados: {len(vectors)}, dimensiones: {len(vectors[0])}")

    # Matriz de similitud
    t_vectors = torch.tensor(vectors)
    sim_matrix = torch.nn.functional.cosine_similarity(
        t_vectors.unsqueeze(1), t_vectors.unsqueeze(0), dim=2
    )

    print("\n📊 Matriz de Similitud:")
    print(f"{'':>6}", end="")
    for i in range(len(texts)):
        print(f"  T{i}   ", end="")
    print()

    for i in range(len(texts)):
        print(f"T{i}  ", end="")
        for j in range(len(texts)):
            score = sim_matrix[i][j].item()
            marker = " *" if i != j and score > 0.7 else "  "
            print(f" {score:.3f}{marker}", end="")
        print(f"  ← {texts[i][:35]}...")

    # Los textos del gato y perro (ambos animales) deben ser más similares
    # que gato vs programación
    sim_gato_perro = sim_matrix[0][1].item()
    sim_gato_python = sim_matrix[0][2].item()

    print(f"\n🐱🐶 Gato-Perro: {sim_gato_perro:.4f}")
    print(f"🐱💻 Gato-Python: {sim_gato_python:.4f}")

    if sim_gato_perro > sim_gato_python:
        print("✅ Correcto: Textos semánticamente similares tienen mayor score")
    else:
        print("❌ Inesperado: La similitud semántica no es coherente")


def test_api_endpoints():
    """
    Prueba los endpoints REST del gateway.
    Requiere que el servidor esté corriendo.
    """
    import requests

    BASE_URL = os.getenv("LLM_GATEWAY_URL", "http://localhost:8100")
    print("\n" + "=" * 60)
    print(f"🧪 PRUEBA 3: API Endpoints ({BASE_URL})")
    print("=" * 60)

    # Test 1: Embedding de texto individual
    print("\n📍 POST /v1/embeddings/text")
    try:
        resp = requests.post(f"{BASE_URL}/v1/embeddings/text", json={
            "text": "El gato duerme en el sofá",
            "normalize": True
        }, timeout=30)
        data = resp.json()

        if resp.status_code == 200:
            print(f"   ✅ Status: {resp.status_code}")
            print(f"   📐 Dimensiones: {data['dimensions']}")
            print(f"   🏷️ Modelo: {data['model']}")
            print(f"   📊 Primeros 5 valores: {data['embedding'][:5]}")
        else:
            print(f"   ❌ Error ({resp.status_code}): {data}")
    except requests.ConnectionError:
        print(f"   ⚠️ No se pudo conectar a {BASE_URL}. ¿Está el servidor corriendo?")
        return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # Test 2: Batch embeddings
    print("\n📍 POST /v1/embeddings/text/batch")
    try:
        resp = requests.post(f"{BASE_URL}/v1/embeddings/text/batch", json={
            "texts": [
                "Primera frase de prueba",
                "Segunda frase de prueba",
                "Tercera frase completamente diferente"
            ],
            "normalize": True
        }, timeout=30)
        data = resp.json()

        if resp.status_code == 200:
            print(f"   ✅ Status: {resp.status_code}")
            print(f"   📦 Total vectores: {data['total']}")
            print(f"   📐 Dimensiones: {len(data['data'][0]['embedding'])}")
        else:
            print(f"   ❌ Error ({resp.status_code}): {data}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 3: Info de modelos
    print("\n📍 GET /v1/embeddings/models")
    try:
        resp = requests.get(f"{BASE_URL}/v1/embeddings/models", timeout=10)
        data = resp.json()

        if resp.status_code == 200:
            print(f"   ✅ Status: {resp.status_code}")
            print(f"   🖥️ Dispositivo: {data['device']}")
            for modality, model_info in data['models'].items():
                print(f"   📌 {modality}: {model_info['model_id']} ({model_info['dimensions']}d)")
        else:
            print(f"   ❌ Error ({resp.status_code}): {data}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pruebas de embeddings del LLM Gateway")
    parser.add_argument("--api", action="store_true", help="Probar endpoints REST (requiere servidor)")
    parser.add_argument("--skip-local", action="store_true", help="Saltar pruebas locales (solo API)")
    args = parser.parse_args()

    print("🚀 Test de Embeddings - LLM Gateway")
    print("=" * 60)

    if not args.skip_local:
        test_local_embedding_quality()
        test_batch_embeddings()

    if args.api:
        test_api_endpoints()

    print("\n" + "=" * 60)
    print("✅ Todas las pruebas completadas")
    print("=" * 60)
