import redis
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path

class SessionManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Maneja sesiones de conversación con Redis + filesystem.
        
        Redis almacena:
        - Historial de mensajes
        - Metadata de archivos generados
        - Imágenes de referencia (base64)
        
        Filesystem almacena:
        - Videos/imágenes generados temporalmente
        """
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.files_dir = Path("temp_files")
        self.files_dir.mkdir(exist_ok=True)
        
        # TTL por tipo de dato
        self.SESSION_TTL = int(timedelta(hours=24).total_seconds())  # 24h
        self.FILE_TTL = int(timedelta(hours=2).total_seconds())      # 2h
    
    def _get_session_key(self, conversation_uuid: str) -> str:
        """Genera key de Redis para la sesión."""
        return f"session:{conversation_uuid}"
    
    def _get_files_key(self, conversation_uuid: str) -> str:
        """Genera key de Redis para archivos de la sesión."""
        return f"files:{conversation_uuid}"
    
    def _get_refs_key(self, conversation_uuid: str) -> str:
        """Genera key para imágenes de referencia."""
        return f"refs:{conversation_uuid}"
    
    async def create_session(self, conversation_uuid: str) -> Dict:
        """Crea una nueva sesión."""
        session_key = self._get_session_key(conversation_uuid)
        
        session_data = {
            "uuid": conversation_uuid,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        }
        
        self.redis_client.setex(
            session_key,
            self.SESSION_TTL,
            json.dumps(session_data)
        )
        
        return session_data
    
    async def get_session(self, conversation_uuid: str) -> Optional[Dict]:
        """Obtiene datos de la sesión."""
        session_key = self._get_session_key(conversation_uuid)
        data = self.redis_client.get(session_key)
        
        if data:
            return json.loads(data)
        return None
    
    async def add_message(self, conversation_uuid: str, role: str, content: str):
        """Agrega un mensaje al historial."""
        session = await self.get_session(conversation_uuid)
        
        if not session:
            session = await self.create_session(conversation_uuid)
        
        session["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        session_key = self._get_session_key(conversation_uuid)
        self.redis_client.setex(
            session_key,
            self.SESSION_TTL,
            json.dumps(session)
        )
    
    async def save_generated_file(
        self, 
        conversation_uuid: str, 
        file_url: str,
        file_type: str,
        metadata: Dict = None
    ) -> Dict:
        """
        Guarda metadata de archivo generado en Redis.
        Retorna info del archivo.
        """
        file_id = str(uuid.uuid4())
        
        # Guardar metadata en Redis
        file_info = {
            "file_id": file_id,
            "url": file_url,
            "type": file_type,
            "created_at": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        files_key = self._get_files_key(conversation_uuid)
        print(f"DEBUG [session_manager]: Saving file to Redis key='{files_key}', url='{file_url}', type='{file_type}'")
        self.redis_client.lpush(files_key, json.dumps(file_info))
        self.redis_client.expire(files_key, self.FILE_TTL)
        
        # Verify it was saved
        count = self.redis_client.llen(files_key)
        print(f"DEBUG [session_manager]: File saved. Total files in '{files_key}': {count}")
        
        return file_info
    
    async def get_pending_files(self, conversation_uuid: str) -> List[Dict]:
        """Obtiene archivos pendientes de enviar al usuario."""
        files_key = self._get_files_key(conversation_uuid)
        print(f"DEBUG [session_manager]: Getting pending files from Redis key='{files_key}'")
        file_list = self.redis_client.lrange(files_key, 0, -1)
        print(f"DEBUG [session_manager]: Found {len(file_list)} files in Redis")
        
        return [json.loads(f) for f in file_list]
    
    async def clear_sent_files(self, conversation_uuid: str):
        """Elimina archivos ya enviados (limpieza)."""
        files_key = self._get_files_key(conversation_uuid)
        self.redis_client.delete(files_key)
    
    async def save_reference_files(self, conversation_uuid: str, files_data: List[Dict]):
        """Guarda archivos de referencia como URLs (persisten en la sesión)."""
        refs_key = self._get_refs_key(conversation_uuid)
        self.redis_client.setex(
            refs_key,
            self.SESSION_TTL,
            json.dumps(files_data)
        )
    
    async def get_reference_files(self, conversation_uuid: str) -> List[Dict]:
        """Obtiene archivos de referencia de la sesión."""
        refs_key = self._get_refs_key(conversation_uuid)
        data = self.redis_client.get(refs_key)
        
        if data:
            return json.loads(data)
        return []
    
    # Mantener compatibilidad con métodos antiguos
    async def save_reference_images(self, conversation_uuid: str, images_b64: List[str]):
        """Legacy method - ahora guarda URLs."""
        files_data = [{"url": img, "type": "image"} for img in images_b64]
        await self.save_reference_files(conversation_uuid, files_data)
    
    async def get_reference_images(self, conversation_uuid: str) -> List[str]:
        """Legacy method - retorna URLs."""
        files = await self.get_reference_files(conversation_uuid)
        return [f["url"] for f in files] if files else []
    
    async def delete_session(self, conversation_uuid: str):
        """Elimina una sesión completa."""
        session_key = self._get_session_key(conversation_uuid)
        files_key = self._get_files_key(conversation_uuid)
        refs_key = self._get_refs_key(conversation_uuid)
        
        self.redis_client.delete(session_key, files_key, refs_key)


# Singleton
_session_manager: Optional[SessionManager] = None

def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _session_manager = SessionManager(redis_url)
    return _session_manager
