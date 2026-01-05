"""
Script de prueba CORREGIDO para el endpoint multimodal
Asegura formato correcto de multipart/form-data
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
    
    # Datos del formulario
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


def test_vision_with_image(image_path: str = None):
    """Prueba vision con imagen adjunta"""
    print("\n" + "="*60)
    print("🖼️  Test 2: Vision con Imagen")
    print("="*60)
    
    if not image_path or not Path(image_path).exists():
        print("⚠️  No se proporcionó imagen válida")
        print("   Llamar con: test_vision_with_image('ruta/imagen.jpg')")
        return
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe detalladamente lo que ves en esta imagen"},
                {"type": "image", "file_index": 0}
            ]
        }
    ]
    
    # Preparar datos del formulario
    data = {
        "task": "vision",
        "privacy_mode": "flexible",
        "messages": json.dumps(messages),
        "temperature": 0.7
    }
    
    # Abrir y enviar archivo
    try:
        with open(image_path, 'rb') as f:
            # Importante: usar 'files' como lista de tuplas
            files = [('files', (Path(image_path).name, f, 'image/jpeg'))]
            response = requests.post(API_ENDPOINT, data=data, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Respuesta recibida:")
            print(f"   Modelo: {result.get('model')}")
            print(f"   Respuesta: {result['choices'][0]['message']['content'][:300]}...")
        else:
            print(f"❌ Error {response.status_code}")
            print(f"   Detalle: {response.text[:500]}")
    
    except Exception as e:
        print(f"❌ Excepción: {str(e)}")


def test_multimodal_conversation():
    """Prueba conversación multimodal con múltiples mensajes"""
    print("\n" + "="*60)
    print("🎭 Test 3: Conversación Multimodal")
    print("="*60)
    
    messages = [
        {
            "role": "system",
            "content": "Eres un asistente útil y conciso"
        },
        {
            "role": "user",
            "content": "Hola, ¿cómo estás?"
        },
        {
            "role": "assistant",
            "content": "¡Hola! Estoy bien, gracias. ¿En qué puedo ayudarte?"
        },
        {
            "role": "user",
            "content": "Explícame qué es un embedding en 2 frases"
        }
    ]
    
    data = {
        "task": "chat",
        "privacy_mode": "strict",
        "messages": json.dumps(messages),
        "temperature": 0.5,
        "max_tokens": 100
    }
    
    response = requests.post(API_ENDPOINT, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Respuesta recibida:")
        print(f"   Modelo: {result.get('model')}")
        usage = result.get('usage', {})
        if usage:
            print(f"   Tokens usados: {usage.get('total_tokens', 'N/A')}")
        print(f"   Respuesta: {result['choices'][0]['message']['content']}")
    else:
        print(f"❌ Error {response.status_code}")
        print(f"   Detalle: {response.text[:500]}")


def test_error_handling():
    """Prueba manejo de errores"""
    print("\n" + "="*60)
    print("🐛 Test 4: Manejo de Errores")
    print("="*60)
    
    # Test 1: Task inválido
    print("\n  Test 4.1: Task inválido")
    data = {
        "task": "invalid_task",
        "privacy_mode": "strict",
        "messages": json.dumps([{"role": "user", "content": "test"}])
    }
    response = requests.post(API_ENDPOINT, data=data)
    if response.status_code == 400:
        print("  ✅ Error 400 correctamente capturado")
        print(f"     {response.json().get('detail', '')[:80]}")
    else:
        print(f"  ❌ Esperaba 400, recibió {response.status_code}")
    
    # Test 2: Messages JSON inválido
    print("\n  Test 4.2: JSON inválido")
    data = {
        "task": "chat",
        "privacy_mode": "strict",
        "messages": "not a valid json"
    }
    response = requests.post(API_ENDPOINT, data=data)
    if response.status_code == 400:
        print("  ✅ Error 400 correctamente capturado")
        print(f"     {response.json().get('detail', '')[:80]}")
    else:
        print(f"  ❌ Esperaba 400, recibió {response.status_code}")
    
    # Test 3: file_index fuera de rango
    print("\n  Test 4.3: file_index inválido")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "test"},
                {"type": "image", "file_index": 99}  # No hay archivo
            ]
        }
    ]
    data = {
        "task": "vision",
        "privacy_mode": "flexible",
        "messages": json.dumps(messages)
    }
    response = requests.post(API_ENDPOINT, data=data)
    if response.status_code == 400:
        print("  ✅ Error 400 correctamente capturado")
        print(f"     {response.json().get('detail', '')[:80]}")
    else:
        print(f"  ❌ Esperaba 400, recibió {response.status_code}")


def main():
    """Ejecuta todas las pruebas"""
    print("\n" + "="*60)
    print("🚀 PRUEBAS DE ENDPOINT MULTIMODAL")
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
        
        # Ejecutar pruebas
        test_simple_chat()
        test_multimodal_conversation()
        test_error_handling()
        
        # Pruebas con archivos (requieren archivos reales)
        print("\n" + "="*60)
        print("ℹ️  Pruebas con archivos multimedia")
        print("="*60)
        print("Para probar con archivos, ejecuta manualmente:")
        print("\n  from test_multimodal_chat import *")
        print("  test_vision_with_image('imagen.jpg')")
        
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
