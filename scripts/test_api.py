"""
Script de prueba para verificar el API Gateway
"""
import sys
import os
import asyncio

# Add project root to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.router import model_router
from services.llm_client import call_llm
from config import settings

async def test_router():
    """Prueba el router de modelos"""
    print("=" * 60)
    print("PRUEBA 1: Model Router")
    print("=" * 60)
    
    # Test 1: Chat strict (local)
    model = model_router.select_model("chat", "strict")
    print(f"✓ Chat Strict: {model}")
    assert "ollama" in model.lower()
    
    # Test 2: Vision flexible (cloud)
    model = model_router.select_model("vision", "flexible")
    print(f"✓ Vision Flexible: {model}")
    assert "gemini" in model.lower()
    
    # Test 3: OCR strict (local)
    model = model_router.select_model("ocr", "strict")
    print(f"✓ OCR Strict: {model}")
    assert "ollama" in model.lower()
    
    print("\n✅ Model Router: PASSED\n")


async def test_api_structure():
    """Verifica la estructura de la API"""
    print("=" * 60)
    print("PRUEBA 2: API Structure")
    print("=" * 60)
    
    from main import app
    
    # Verificar que la app existe
    print(f"✓ FastAPI App: {app.title}")
    
    # Verificar rutas
    routes = [route.path for route in app.routes]
    print(f"✓ Rutas disponibles: {len(routes)}")
    
    assert "/health" in routes
    assert "/v1/chat/completions" in routes
    
    print("\n✅ API Structure: PASSED\n")


def test_config():
    """Prueba la configuración"""
    print("=" * 60)
    print("PRUEBA 3: Configuration")
    print("=" * 60)
    
    print(f"✓ Ollama URL: {settings.ollama_base_url}")
    print(f"✓ Gemini API Key configurada: {'Sí' if settings.gemini_api_key else 'No (opcional)'}")
    print(f"✓ Log Level: {settings.log_level}")
    
    # Verificar MODEL_ROUTER
    assert "chat" in settings.MODEL_ROUTER
    assert "vision" in settings.MODEL_ROUTER
    assert "ocr" in settings.MODEL_ROUTER
    assert "embedding" in settings.MODEL_ROUTER
    
    print("\n✅ Configuration: PASSED\n")


async def main():
    """Ejecuta todas las pruebas"""
    print("\n🧪 INICIANDO PRUEBAS DEL LLM GATEWAY API\n")
    
    try:
        test_config()
        await test_router()
        await test_api_structure()
        
        print("=" * 60)
        print("🎉 TODAS LAS PRUEBAS PASARON")
        print("=" * 60)
        print("\nPara iniciar el servidor ejecuta:")
        print("  uvicorn main:app --reload --port 8765")
        print("\nLuego accede a:")
        print("  http://localhost:8765/docs")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
