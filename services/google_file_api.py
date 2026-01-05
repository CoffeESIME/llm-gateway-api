"""
Cliente para Google File API
Maneja la subida de archivos grandes a servidores de Google
"""
import os
import requests
import logging
from typing import Optional, Dict, Any
from config import settings

logger = logging.getLogger(__name__)

# Endpoints de Google File API
GOOGLE_FILE_API_BASE = "https://generativelanguage.googleapis.com"
UPLOAD_ENDPOINT = f"{GOOGLE_FILE_API_BASE}/upload/v1beta/files"
FILES_ENDPOINT = f"{GOOGLE_FILE_API_BASE}/v1beta/files"


class GoogleFileAPIError(Exception):
    """Error personalizado para Google File API"""
    pass


class GoogleFileAPI:
    """Cliente para interactuar con Google File API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el cliente de Google File API
        
        Args:
            api_key: API key de Google (usa settings si no se proporciona)
        """
        self.api_key = api_key or settings.gemini_api_key
        if not self.api_key:
            raise GoogleFileAPIError(
                "GEMINI_API_KEY no configurada. "
                "Necesaria para usar Google File API."
            )
    
    def upload_file(
        self, 
        file_bytes: bytes, 
        filename: str, 
        mime_type: str
    ) -> str:
        """
        Sube un archivo a Google File API
        
        Args:
            file_bytes: Contenido del archivo en bytes
            filename: Nombre del archivo
            mime_type: Tipo MIME (ej: "audio/mp3", "video/mp4")
        
        Returns:
            URI del archivo en Google (ej: https://generativelanguage.googleapis.com/v1beta/files/abc123)
        
        Raises:
            GoogleFileAPIError: Si hay error en la subida
        """
        try:
            logger.info(f"📤 Subiendo archivo a Google File API: {filename} ({len(file_bytes) / 1024 / 1024:.1f}MB)")
            
            # Preparar headers
            headers = {
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "upload, finalize",
                "X-Goog-Upload-Header-Content-Length": str(len(file_bytes)),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": mime_type
            }
            
            # URL con API key
            url = f"{UPLOAD_ENDPOINT}?key={self.api_key}"
            
            # Subir archivo (resumable upload en un solo paso)
            response = requests.post(
                url,
                headers=headers,
                data=file_bytes,
                timeout=120  # 2 minutos timeout para archivos grandes
            )
            
            # Verificar respuesta
            if response.status_code not in [200, 201]:
                error_detail = response.text
                logger.error(f"❌ Error subiendo a Google File API: {error_detail}")
                
                # Errores específicos
                if response.status_code == 401:
                    raise GoogleFileAPIError(
                        "Error de autenticación con Google File API. "
                        "Verifica tu GEMINI_API_KEY."
                    )
                elif response.status_code == 403:
                    raise GoogleFileAPIError(
                        "Acceso denegado. Verifica que tu API key tenga permisos "
                        "para File API."
                    )
                elif response.status_code == 429:
                    raise GoogleFileAPIError(
                        "Límite de rate excedido en Google File API. "
                        "Intenta más tarde."
                    )
                else:
                    raise GoogleFileAPIError(
                        f"Error subiendo archivo: {response.status_code} - {error_detail}"
                    )
            
            # Extraer URI del archivo
            result = response.json()
            file_data = result.get("file", {})
            file_uri = file_data.get("uri")
            
            if not file_uri:
                raise GoogleFileAPIError(
                    f"No se pudo obtener URI del archivo. Respuesta: {result}"
                )
            
            logger.info(f"✅ Archivo subido exitosamente: {file_uri}")
            return file_uri
            
        except requests.exceptions.Timeout:
            raise GoogleFileAPIError(
                "Timeout subiendo archivo a Google. "
                "El archivo podría ser demasiado grande o la conexión lenta."
            )
        except requests.exceptions.RequestException as e:
            raise GoogleFileAPIError(f"Error de red subiendo archivo: {str(e)}")
        except Exception as e:
            if isinstance(e, GoogleFileAPIError):
                raise
            raise GoogleFileAPIError(f"Error inesperado: {str(e)}")
    
    def delete_file(self, file_uri: str) -> bool:
        """
        Elimina un archivo de Google File API
        
        Args:
            file_uri: URI completa del archivo
        
        Returns:
            True si se eliminó exitosamente
        
        Raises:
            GoogleFileAPIError: Si hay error eliminando
        """
        try:
            # Extraer el nombre del archivo de la URI
            # URI: https://generativelanguage.googleapis.com/v1beta/files/abc123
            file_name = file_uri.split("/files/")[-1]
            
            url = f"{FILES_ENDPOINT}/{file_name}?key={self.api_key}"
            
            logger.info(f"🗑️  Eliminando archivo: {file_name}")
            
            response = requests.delete(url, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"✅ Archivo eliminado: {file_name}")
                return True
            elif response.status_code == 404:
                logger.warning(f"⚠️  Archivo no encontrado (ya eliminado?): {file_name}")
                return True  # Consideramos éxito si ya no existe
            else:
                logger.error(f"❌ Error eliminando archivo: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error eliminando archivo: {str(e)}")
            return False
    
    def get_file_info(self, file_uri: str) -> Dict[str, Any]:
        """
        Obtiene información de un archivo
        
        Args:
            file_uri: URI completa del archivo
        
        Returns:
            Diccionario con info del archivo
        
        Raises:
            GoogleFileAPIError: Si hay error obteniendo info
        """
        try:
            file_name = file_uri.split("/files/")[-1]
            url = f"{FILES_ENDPOINT}/{file_name}?key={self.api_key}"
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise GoogleFileAPIError(
                    f"Error obteniendo info del archivo: {response.text}"
                )
                
        except Exception as e:
            if isinstance(e, GoogleFileAPIError):
                raise
            raise GoogleFileAPIError(f"Error inesperado: {str(e)}")


# Instancia global (singleton)
google_file_api = GoogleFileAPI()
