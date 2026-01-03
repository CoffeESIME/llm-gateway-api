"""
Servicio de routing de modelos
Selecciona el modelo apropiado basado en task y privacy_mode
"""
import logging
from typing import Optional
from config import settings

logger = logging.getLogger(__name__)


class ModelRouter:
    """Router de modelos basado en tarea y modo de privacidad"""
    
    def __init__(self):
        self.model_map = settings.MODEL_ROUTER
    
    def select_model(
        self,
        task: str,
        privacy_mode: str,
        override_model: Optional[str] = None
    ) -> str:
        """
        Selecciona el modelo apropiado basado en los parámetros
        
        Args:
            task: Tipo de tarea (chat, vision, ocr, embedding)
            privacy_mode: Modo de privacidad (strict, flexible)
            override_model: Modelo específico para override manual (opcional)
        
        Returns:
            str: Nombre del modelo a usar (formato litellm: provider/model-name)
        
        Raises:
            ValueError: Si la combinación de task/privacy_mode no existe
        """
        # Si hay override manual, usarlo
        if override_model:
            logger.info(f"🎯 Usando modelo manual override: {override_model}")
            return override_model
        
        # Validar que la tarea existe
        if task not in self.model_map:
            available_tasks = ", ".join(self.model_map.keys())
            raise ValueError(
                f"Task '{task}' no válido. Tareas disponibles: {available_tasks}"
            )
        
        # Validar que el modo existe para esta tarea
        if privacy_mode not in self.model_map[task]:
            available_modes = ", ".join(self.model_map[task].keys())
            raise ValueError(
                f"Privacy mode '{privacy_mode}' no válido para task '{task}'. "
                f"Modos disponibles: {available_modes}"
            )
        
        # Obtener el modelo
        selected_model = self.model_map[task][privacy_mode]
        
        # Logging informativo
        privacy_emoji = "🔒" if privacy_mode == "strict" else "☁️"
        logger.info(
            f"{privacy_emoji} Task: {task} | Privacy: {privacy_mode} | "
            f"Modelo seleccionado: {selected_model}"
        )
        
        return selected_model
    
    def get_available_tasks(self) -> list:
        """Retorna lista de tareas disponibles"""
        return list(self.model_map.keys())
    
    def get_available_modes(self, task: str) -> list:
        """Retorna lista de modos disponibles para una tarea"""
        if task not in self.model_map:
            return []
        return list(self.model_map[task].keys())


# Instancia global del router
model_router = ModelRouter()
