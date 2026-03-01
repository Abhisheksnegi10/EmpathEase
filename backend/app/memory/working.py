"""
Working Memory - In-memory + optional Redis session state management.

Stores:
- Last 20 conversation turns
- Current emotional state
- Context entities (people, topics)
- Therapeutic approach in use

Falls back to in-memory storage when Redis is unavailable.
"""

import json
import logging
from typing import Dict, List, Optional
from uuid import UUID

from app.config import get_settings
from app.schemas.emotion import ConversationTurn, WorkingMemory, FusedEmotionalState

logger = logging.getLogger(__name__)
settings = get_settings()

# In-memory fallback store
_memory_store: Dict[str, str] = {}


async def _redis_or_memory():
    """
    Try to get a Redis client. If it fails (connection refused, etc.),
    return None and callers will use the in-memory _memory_store.
    """
    try:
        from app.db import get_redis
        redis = get_redis()
        # Quick liveness check for real Redis (skip for in-memory mock)
        if hasattr(redis, "ping"):
            await redis.ping()
        return redis
    except Exception:
        return None


class WorkingMemoryManager:
    """
    Manages session working memory in Redis (or in-memory fallback).

    Features:
    - Automatic TTL expiration (30 minutes after last activity)
    - Capped conversation history (last 20 turns)
    - Entity extraction and tracking
    """

    def __init__(self, session_id: UUID):
        self.session_id = session_id
        self._key_prefix = f"session:{session_id}"

    # ---- Low-level get/set with fallback ----

    async def _get(self, key: str) -> Optional[str]:
        redis = await _redis_or_memory()
        if redis:
            return await redis.get(key)
        return _memory_store.get(key)

    async def _setex(self, key: str, ttl: int, value: str) -> None:
        redis = await _redis_or_memory()
        if redis:
            await redis.setex(key, ttl, value)
        else:
            _memory_store[key] = value

    async def _delete(self, key: str) -> None:
        redis = await _redis_or_memory()
        if redis:
            await redis.delete(key)
        else:
            _memory_store.pop(key, None)

    # ---- Public API ----

    async def initialize(self) -> None:
        """Initialize working memory for a new session."""
        initial_state = WorkingMemory(
            session_id=str(self.session_id),
            turns=[],
            context_entities=[],
            current_topic=None,
            therapeutic_approach="person_centered",
        )

        await self._setex(
            f"{self._key_prefix}:state",
            settings.working_memory_ttl,
            initial_state.model_dump_json(),
        )

        logger.info("Working memory initialized for session %s", self.session_id)

    async def add_turn(
        self,
        user_input: str,
        system_response: str,
        emotional_state: Dict,
    ) -> None:
        """Add a conversation turn to working memory."""
        state_json = await self._get(f"{self._key_prefix}:state")
        if not state_json:
            await self.initialize()
            state_json = await self._get(f"{self._key_prefix}:state")

        state = WorkingMemory.model_validate_json(state_json)

        from datetime import datetime

        turn = ConversationTurn(
            turn_id=len(state.turns) + 1,
            timestamp=datetime.utcnow().isoformat(),
            user_input=user_input,
            emotional_state=FusedEmotionalState(**emotional_state),
            system_response=system_response,
        )

        state.turns.append(turn)
        if len(state.turns) > settings.max_conversation_turns:
            state.turns = state.turns[-settings.max_conversation_turns:]

        entities = self._extract_entities(user_input)
        for entity in entities:
            if entity not in state.context_entities:
                state.context_entities.append(entity)

        await self._setex(
            f"{self._key_prefix}:state",
            settings.working_memory_ttl,
            state.model_dump_json(),
        )

    async def get_state(self) -> Optional[WorkingMemory]:
        """Get current working memory state."""
        state_json = await self._get(f"{self._key_prefix}:state")
        if not state_json:
            return None
        return WorkingMemory.model_validate_json(state_json)

    async def get_recent_turns(self, count: int = 5) -> List[ConversationTurn]:
        """Get the most recent conversation turns."""
        state = await self.get_state()
        if not state:
            return []
        return state.turns[-count:]

    async def get_context_for_llm(self) -> str:
        """Get formatted context for LLM prompt."""
        state = await self.get_state()
        if not state:
            return ""

        lines = []
        if state.context_entities:
            lines.append(f"Known entities: {', '.join(state.context_entities)}")
        if state.turns:
            lines.append("\nRecent conversation:")
            for turn in state.turns[-3:]:
                lines.append(f"User: {turn.user_input[:100]}...")
                lines.append(f"System: {turn.system_response[:100]}...")
        return "\n".join(lines)

    async def update_therapeutic_approach(self, approach: str) -> None:
        """Update the therapeutic approach being used."""
        state_json = await self._get(f"{self._key_prefix}:state")
        if not state_json:
            return
        state = WorkingMemory.model_validate_json(state_json)
        state.therapeutic_approach = approach
        await self._setex(
            f"{self._key_prefix}:state",
            settings.working_memory_ttl,
            state.model_dump_json(),
        )

    async def cleanup(self) -> Optional[WorkingMemory]:
        """Clean up working memory (called on session end)."""
        state = await self.get_state()
        await self._delete(f"{self._key_prefix}:state")
        logger.info("Working memory cleaned up for session %s", self.session_id)
        return state

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction (capitalized words)."""
        import re
        words = text.split()
        entities = []
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper():
                clean_word = re.sub(r'[^\w]', '', word)
                if len(clean_word) > 1 and clean_word not in ["I", "I'm", "I've"]:
                    entities.append(clean_word)
        return entities[:5]


# Singleton helper
_managers: Dict[str, WorkingMemoryManager] = {}


def get_working_memory(session_id: UUID) -> WorkingMemoryManager:
    """Get working memory manager for a session."""
    global _managers
    key = str(session_id)
    if key not in _managers:
        _managers[key] = WorkingMemoryManager(session_id)
    return _managers[key]
