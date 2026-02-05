"""
Script de prueba ACTUALIZADO con soporte para Google File API
Incluye tests para archivos grandes
"""
import requests
import json
from pathlib import Path

# Configuración
BASE_URL = "http://localhost:8765"
API_ENDPOINT = f"{BASE_URL}/v1/chat/completions"


def test_simple_chat():
    """Prueba chat simple sin archivos"""
    print("\n" + "="*60)
    print("💬 Test 1: Chat Simple (sin archivos)")
    print("="*60)
    
    messages = [
        {
            "role": "user",
            "content": "¿Cuál es la capital de Francia?"
        }
    ]
    
    data = {
        "task": "chat",
        "privacy_mode": "strict",
        "messages": json.dumps(messages),
        "temperature": 0.7
    }
    
    response = requests.post(API_ENDPOINT, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Respuesta recibida:")
        print(f"   Modelo: {result.get('model')}")
        print(f"   Respuesta: {result['choices'][0]['message']['content'][:200]}...")
    else:
        print(f"❌ Error {response.status_code}")
        print(f"   Detalle: {response.text[:500]}")


def test_small_image_base64(image_path: str = None):
    """Prueba imagen pequeña (< 5MB) - debería usar base64"""
    print("\n" + "="*60)
    print("🖼️  Test 2: Imagen Pequeña (Base64)")
    print("="*60)
    
    if not image_path or not Path(image_path).exists():
        print("⚠️  No se proporcionó imagen válida")
        print("   Llamar con: test_small_image_base64('imagen_pequena.jpg')")
        return
    
    file_size = Path(image_path).stat().st_size
    print(f"   Tamaño: {file_size / 1024:.1f}KB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe esta imagen"},
                {"type": "image", "file_index": 0}
            ]
        }
    ]
    
    data = {
        "task": "vision",
        "privacy_mode": "flexible",
        "messages": json.dumps(messages)
    }
    
    try:
        with open(image_path, 'rb') as f:
            files = [('files', (Path(image_path).name, f, 'image/jpeg'))]
            response = requests.post(API_ENDPOINT, data=data, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Respuesta recibida (Base64):")
            print(f"   Modelo: {result.get('model')}")
            print(f"   Respuesta: {result['choices'][0]['message']['content'][:200]}...")
        else:
            print(f"❌ Error {response.status_code}")
            print(f"   Detalle: {response.text[:500]}")
    
    except Exception as e:
        print(f"❌ Excepción: {str(e)}")


def test_large_file_google_api(file_path: str = None, file_type: str = "audio"):
    """Prueba archivo grande (>= 5MB) con Google File API"""
    print("\n" + "="*60)
    print(f"🎵 Test 3: Archivo Grande (Google File API) - {file_type}")
    print("="*60)
    
    if not file_path or not Path(file_path).exists():
        print("⚠️  No se proporcionó archivo válido")
        print(f"   Llamar con: test_large_file_google_api('archivo.{file_type}')")
        return
    
    file_size = Path(file_path).stat().st_size
    print(f"   Tamaño: {file_size / 1024 / 1024:.1f}MB")
    
    if file_size < 5 * 1024 * 1024:
        print("   ⚠️  Archivo < 5MB - usará base64 en lugar de File API")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Analiza este {file_type}"},
                {"type": file_type, "file_index": 0}
            ]
        }
    ]
    
    data = {
        "task": "vision",
        "privacy_mode": "flexible",  # Requerido para archivos grandes
        "messages": json.dumps(messages)
    }
    
    try:
        content_types = {
            "audio": "audio/mp3",
            "video": "video/mp4",
            "image": "image/jpeg"
        }
        
        with open(file_path, 'rb') as f:
            files = [('files', (Path(file_path).name, f, content_types.get(file_type, "application/octet-stream")))]
            response = requests.post(API_ENDPOINT, data=data, files=files, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Respuesta recibida (Google File API):")
            print(f"   Modelo: {result.get('model')}")
            print(f"   Respuesta: {result['choices'][0]['message']['content'][:300]}...")
        else:
            print(f"❌ Error {response.status_code}")
            print(f"   Detalle: {response.text[:500]}")
    
    except Exception as e:
        print(f"❌ Excepción: {str(e)}")


def test_large_file_strict_mode(file_path: str = None):
    """Prueba archivo grande con privacy_mode=strict (debería fallar con NotImplementedError)"""
    print("\n" + "="*60)
    print("🚫 Test 4: Archivo Grande + Strict Mode (Esperado: Error 501)")
    print("="*60)
    
    if not file_path or not Path(file_path).exists():
        print("⚠️  No se proporcionó archivo válido")
        return
    
    file_size = Path(file_path).stat().st_size
    print(f"   Tamaño: {file_size / 1024 / 1024:.1f}MB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analiza esto"},
                {"type": "audio", "file_index": 0}
            ]
        }
    ]
    
    data = {
        "task": "chat",
        "privacy_mode": "strict",  # Modo estricto con archivo grande
        "messages": json.dumps(messages)
    }
    
    try:
        with open(file_path, 'rb') as f:
            files = [('files', (Path(file_path).name, f, 'audio/mp3'))]
            response = requests.post(API_ENDPOINT, data=data, files=files)
        
        if response.status_code == 501:
            print("✅ Error 501 correctamente recibido (NotImplemented)")
            detail = response.json().get('detail', '')
            if 'TODO' in detail:
                print("   ✓ Mensaje incluye TODOs para chunking local")
                print(f"   Detalle: {detail[:200]}...")
        else:
            print(f"❌ Esperaba 501, recibió {response.status_code}")
            print(f"   Detalle: {response.text[:500]}")
    
    except Exception as e:
        print(f"❌ Excepción: {str(e)}")


def main():
    """Ejecuta todas las pruebas"""
    print("\n" + "="*60)
    print("🚀 PRUEBAS DE GOOGLE FILE API")
    print("="*60)
    print(f"Endpoint: {API_ENDPOINT}")
    
    try:
        # Verificar conexión
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print("❌ Error: API no está respondiendo")
            print("   Inicia el servidor: python main.py")
            return
        
        print("✅ API conectada correctamente\n")
        
        # Ejecutar pruebas básicas
        test_simple_chat()
        
        # Pruebas con archivos (requieren archivos reales)
        print("\n" + "="*60)
        print("ℹ️  Pruebas con archivos multimedia")
        print("="*60)
        print("\nPara probar con archivos, ejecuta manualmente:")
        print("\n  from test_multimodal_chat import *")
        print("\n  # Imagen pequeña (< 5MB) - usa base64")
        print("  test_small_image_base64('imagen.jpg')")
        print("\n  # Audio grande (>= 5MB) - usa Google File API")
        print("  test_large_file_google_api('audio_grande.mp3', 'audio')")
        print("\n  # Archivo grande + strict mode - debería fallar")
        print("  test_large_file_strict_mode('audio_grande.mp3')")
        
        print("\n" + "="*60)
        print("✅ PRUEBAS COMPLETADAS")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: No se pudo conectar a la API")
        print("   Inicia el servidor: python main.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    main()
