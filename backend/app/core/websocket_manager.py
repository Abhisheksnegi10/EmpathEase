"""
WebSocket connection manager for handling multiple concurrent connections.
"""

import logging
from typing import Dict, List, Set
from uuid import UUID

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for therapy sessions.
    
    Supports:
    - Multiple connections per session (future: therapist view)
    - Connection tracking by user and session
    - Broadcast to session participants
    """

    def __init__(self):
        # session_id -> set of (websocket, user_id)
        self._sessions: Dict[UUID, Set[tuple]] = {}
        # user_id -> websocket (for direct messaging)
        self._users: Dict[str, WebSocket] = {}

    async def connect(
        self,
        websocket: WebSocket,
        session_id: UUID,
        user_id: str,
    ) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        
        # Add to session set
        if session_id not in self._sessions:
            self._sessions[session_id] = set()
        self._sessions[session_id].add((websocket, user_id))
        
        # Add to user map
        self._users[user_id] = websocket
        
        logger.debug(f"Connected: user={user_id}, session={session_id}")

    def disconnect(self, session_id: UUID, user_id: str) -> None:
        """Remove a WebSocket connection."""
        # Remove from session set
        if session_id in self._sessions:
            to_remove = None
            for conn in self._sessions[session_id]:
                if conn[1] == user_id:
                    to_remove = conn
                    break
            if to_remove:
                self._sessions[session_id].discard(to_remove)
            
            # Clean up empty sessions
            if not self._sessions[session_id]:
                del self._sessions[session_id]
        
        # Remove from user map
        if user_id in self._users:
            del self._users[user_id]
        
        logger.debug(f"Disconnected: user={user_id}, session={session_id}")

    async def send_to_user(self, user_id: str, message: dict) -> bool:
        """Send message to a specific user."""
        websocket = self._users.get(user_id)
        if websocket:
            try:
                await websocket.send_json(message)
                return True
            except Exception as e:
                logger.error(f"Failed to send to user {user_id}: {e}")
                return False
        return False

    async def broadcast_to_session(
        self,
        session_id: UUID,
        message: dict,
        exclude_user: str = None,
    ) -> None:
        """Broadcast message to all connections in a session."""
        if session_id not in self._sessions:
            return
        
        for websocket, user_id in self._sessions[session_id]:
            if exclude_user and user_id == exclude_user:
                continue
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast failed for user {user_id}: {e}")

    def get_session_users(self, session_id: UUID) -> List[str]:
        """Get list of user IDs connected to a session."""
        if session_id not in self._sessions:
            return []
        return [user_id for _, user_id in self._sessions[session_id]]

    def get_active_sessions(self) -> List[UUID]:
        """Get list of active session IDs."""
        return list(self._sessions.keys())

    def is_connected(self, user_id: str) -> bool:
        """Check if a user is currently connected."""
        return user_id in self._users
