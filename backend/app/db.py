"""
Database and cache connection helpers.

Provides:
- get_redis()  → async Redis client (falls back to in-memory mock)
- get_db()     → async DB session (placeholder — not yet configured)
"""

import logging
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Redis (with in-memory fallback)
# ---------------------------------------------------------------------------

_redis_client = None
_redis_init_attempted = False


class _InMemoryRedis:
    """
    Minimal async-Redis-compatible mock backed by a plain dict.
    Good enough for single-process demo sessions.
    """

    def __init__(self):
        self._store: dict[str, tuple] = {}  # key → (value, ttl)

    async def get(self, key: str) -> Optional[str]:
        item = self._store.get(key)
        return item[0] if item else None

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = (value, ttl)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def close(self) -> None:
        self._store.clear()


def get_redis():
    """
    Get an async-Redis-compatible client.

    Tries real Redis first; if unavailable, returns an in-memory mock
    so the app still works for demo/development without Redis running.
    """
    global _redis_client, _redis_init_attempted

    if _redis_client is not None:
        return _redis_client

    if not _redis_init_attempted:
        _redis_init_attempted = True
        try:
            import redis.asyncio as aioredis

            _redis_client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
            logger.info("Redis connected at %s", settings.redis_url)
            return _redis_client
        except Exception as e:
            logger.warning(
                "Redis unavailable (%s) — using in-memory fallback", e
            )

    # Fallback
    _redis_client = _InMemoryRedis()
    logger.info("Using in-memory Redis mock")
    return _redis_client


# ---------------------------------------------------------------------------
# Database (placeholder — auth routes disabled)
# ---------------------------------------------------------------------------

async def get_db():
    """
    Async DB session generator.

    TODO: Wire up SQLAlchemy AsyncSession once a database is configured
          for the auth / user-profile features.
    """
    raise NotImplementedError(
        "Database not configured. Enable auth routes after setting up "
        "PostgreSQL / SQLite and creating the User ORM model."
    )
    yield  # noqa: unreachable — keeps this a valid async generator
