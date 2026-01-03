"""
LLM Gateway API - FastAPI Application
Router inteligente entre modelos locales (Ollama) y cloud (Gemini)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from config import settings
from routers import chat

# Configurar logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="LLM Gateway API",
    description="API Gateway inteligente que enruta peticiones entre modelos locales (Ollama) y cloud (Gemini) basado en privacidad",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(chat.router, prefix="/v1", tags=["Chat Completions"])


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    logger.info("🚀 LLM Gateway API iniciando...")
    logger.info(f"📡 Ollama URL: {settings.ollama_base_url}")
    logger.info(f"🔑 Gemini API configurada: {'✓' if settings.gemini_api_key else '✗'}")
    logger.info("📋 Modelos configurados:")
    for task, modes in settings.MODEL_ROUTER.items():
        logger.info(f"  {task}:")
        for mode, model in modes.items():
            logger.info(f"    - {mode}: {model}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
