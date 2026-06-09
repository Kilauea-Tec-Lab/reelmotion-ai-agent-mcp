import redis.asyncio as aioredis
from redis import exceptions as redis_exceptions
import json
import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Maximum messages kept per session to prevent unbounded growth
MAX_HISTORY_MESSAGES = 50


class SessionManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Manages conversation sessions with Redis.

        Redis stores:
        - Message history (capped at MAX_HISTORY_MESSAGES)
        - Generated file metadata
        - Reference file URLs
        - Pending actions awaiting confirmation
        """
        self.redis_client = aioredis.from_url(redis_url, decode_responses=True)

        # TTL per data type
        self.SESSION_TTL = int(timedelta(hours=24).total_seconds())
        self.FILE_TTL = int(timedelta(hours=2).total_seconds())

    def _get_session_key(self, conversation_uuid: str) -> str:
        return f"session:{conversation_uuid}"

    def _get_files_key(self, conversation_uuid: str) -> str:
        return f"files:{conversation_uuid}"

    def _get_refs_key(self, conversation_uuid: str) -> str:
        return f"refs:{conversation_uuid}"

    def _get_pending_action_key(self, conversation_uuid: str) -> str:
        return f"pending_action:{conversation_uuid}"

    def _get_just_generated_key(self, conversation_uuid: str) -> str:
        return f"just_generated:{conversation_uuid}"

    async def create_session(self, conversation_uuid: str) -> Dict:
        """Create a new session."""
        session_key = self._get_session_key(conversation_uuid)

        session_data = {
            "uuid": conversation_uuid,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        }

        await self.redis_client.setex(
            session_key,
            self.SESSION_TTL,
            json.dumps(session_data),
        )

        return session_data

    async def get_session(self, conversation_uuid: str) -> Optional[Dict]:
        """Get session data."""
        session_key = self._get_session_key(conversation_uuid)
        data = await self.redis_client.get(session_key)

        if data:
            return json.loads(data)
        return None

    async def add_message(self, conversation_uuid: str, role: str, content: str):
        """Append a message to history, keeping at most MAX_HISTORY_MESSAGES."""
        session = await self.get_session(conversation_uuid)

        if not session:
            session = await self.create_session(conversation_uuid)

        session["messages"].append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Truncate to prevent unbounded growth
        if len(session["messages"]) > MAX_HISTORY_MESSAGES:
            session["messages"] = session["messages"][-MAX_HISTORY_MESSAGES:]

        session_key = self._get_session_key(conversation_uuid)
        await self.redis_client.setex(
            session_key,
            self.SESSION_TTL,
            json.dumps(session),
        )

    async def save_generated_file(
        self,
        conversation_uuid: str,
        file_url: str,
        file_type: str,
        metadata: Dict = None,
    ) -> Dict:
        """Save generated file metadata to Redis and return file info."""
        file_id = str(uuid.uuid4())

        file_info = {
            "file_id": file_id,
            "url": file_url,
            "type": file_type,
            "created_at": datetime.now().isoformat(),
            **(metadata or {}),
        }

        files_key = self._get_files_key(conversation_uuid)
        logger.debug("Saving file to Redis key='%s', url='%s', type='%s'", files_key, file_url, file_type)
        await self.redis_client.lpush(files_key, json.dumps(file_info))
        await self.redis_client.expire(files_key, self.FILE_TTL)

        count = await self.redis_client.llen(files_key)
        logger.debug("File saved. Total files in '%s': %d", files_key, count)

        return file_info

    async def get_pending_files(self, conversation_uuid: str) -> List[Dict]:
        """Get pending files to send to the user."""
        files_key = self._get_files_key(conversation_uuid)
        logger.debug("Getting pending files from Redis key='%s'", files_key)
        file_list = await self.redis_client.lrange(files_key, 0, -1)
        logger.debug("Found %d files in Redis", len(file_list))

        return [json.loads(f) for f in file_list]

    async def clear_sent_files(self, conversation_uuid: str):
        """Delete already-sent files (cleanup)."""
        files_key = self._get_files_key(conversation_uuid)
        await self.redis_client.delete(files_key)

    async def save_reference_files(self, conversation_uuid: str, files_data: List[Dict]):
        """Save reference file URLs (persist for the session)."""
        refs_key = self._get_refs_key(conversation_uuid)
        await self.redis_client.setex(
            refs_key,
            self.SESSION_TTL,
            json.dumps(files_data),
        )

    async def get_reference_files(self, conversation_uuid: str) -> List[Dict]:
        """Get reference files for the session."""
        refs_key = self._get_refs_key(conversation_uuid)
        data = await self.redis_client.get(refs_key)

        if data:
            return json.loads(data)
        return []

    async def clear_reference_files(self, conversation_uuid: str):
        """Delete reference files for the session."""
        refs_key = self._get_refs_key(conversation_uuid)
        await self.redis_client.delete(refs_key)

    async def save_pending_action(self, conversation_uuid: str, action: Dict):
        """
        Save a pending action awaiting user confirmation.

        action = {
            "function": "generate_video" | "generate_image" | "generate_speech",
            "args": {...},
            "cost_message": "...",
            "timestamp": "..."
        }
        """
        key = self._get_pending_action_key(conversation_uuid)
        action["timestamp"] = datetime.now().isoformat()
        # Short TTL: 5 minutes for confirmation window
        await self.redis_client.setex(key, 300, json.dumps(action))
        logger.debug("Saved pending action '%s' for UUID='%s'", action.get("function"), conversation_uuid)

    async def get_pending_action(self, conversation_uuid: str) -> Optional[Dict]:
        """Get the pending action awaiting confirmation."""
        key = self._get_pending_action_key(conversation_uuid)
        data = await self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None

    async def clear_pending_action(self, conversation_uuid: str):
        """Delete the pending action after execution."""
        key = self._get_pending_action_key(conversation_uuid)
        await self.redis_client.delete(key)
        logger.debug("Cleared pending action for UUID='%s'", conversation_uuid)

    async def claim_pending_action(self, conversation_uuid: str) -> Optional[Dict]:
        """
        Atomically fetch AND delete the pending action (GETDEL).

        Replaces the get-then-delete sequence whose race window allowed two
        concurrent confirmations to execute the same generation twice.
        Returns None if there was no action — i.e. another concurrent request
        already claimed it.
        """
        key = self._get_pending_action_key(conversation_uuid)
        try:
            data = await self.redis_client.getdel(key)
        except (redis_exceptions.ResponseError, AttributeError):
            # GETDEL needs Redis >= 6.2 (the local redis-portable is older).
            # Emulate it with a Lua script, which is still atomic.
            data = await self.redis_client.eval(
                "local v = redis.call('GET', KEYS[1]); "
                "if v then redis.call('DEL', KEYS[1]) end; return v",
                1,
                key,
            )
        if not data:
            return None
        logger.debug("Claimed pending action for UUID='%s'", conversation_uuid)
        return json.loads(data)

    async def set_just_generated(self, conversation_uuid: str):
        """Mark that content was just generated. Expires in 60 seconds."""
        key = self._get_just_generated_key(conversation_uuid)
        await self.redis_client.setex(key, 60, "1")
        logger.debug("Set just_generated flag for UUID='%s'", conversation_uuid)

    async def get_just_generated(self, conversation_uuid: str) -> bool:
        """Check if content was just generated."""
        key = self._get_just_generated_key(conversation_uuid)
        return await self.redis_client.exists(key) > 0

    async def clear_just_generated(self, conversation_uuid: str):
        """Clear the just-generated flag."""
        key = self._get_just_generated_key(conversation_uuid)
        await self.redis_client.delete(key)

    # Legacy compatibility methods
    async def save_reference_images(self, conversation_uuid: str, images_b64: List[str]):
        """Legacy method - now saves URLs."""
        files_data = [{"url": img, "type": "image"} for img in images_b64]
        await self.save_reference_files(conversation_uuid, files_data)

    async def get_reference_images(self, conversation_uuid: str) -> List[str]:
        """Legacy method - returns URLs."""
        files = await self.get_reference_files(conversation_uuid)
        return [f["url"] for f in files] if files else []

    async def delete_session(self, conversation_uuid: str):
        """Delete a complete session."""
        session_key = self._get_session_key(conversation_uuid)
        files_key = self._get_files_key(conversation_uuid)
        refs_key = self._get_refs_key(conversation_uuid)
        pending_key = self._get_pending_action_key(conversation_uuid)

        await self.redis_client.delete(session_key, files_key, refs_key, pending_key)


# Singleton
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _session_manager = SessionManager(redis_url)
    return _session_manager
