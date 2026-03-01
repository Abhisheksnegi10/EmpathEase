"""
Episodic Memory - Qdrant Vector Database Integration.

Stores emotionally significant moments as vector embeddings for:
- Semantic search ("times user felt lonely")
- Similar experience retrieval
- Context-aware therapeutic responses

Uses sentence-transformers for embedding and Qdrant for storage.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from uuid import UUID, uuid4
import hashlib

logger = logging.getLogger(__name__)

# Collection config
COLLECTION_NAME = "emotional_episodes"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
SIMILARITY_THRESHOLD = 0.7


@dataclass
class EmotionalEpisode:
    """Represents an emotionally significant moment."""
    id: str
    user_id: str
    session_id: str
    timestamp: datetime
    summary: str
    dominant_emotion: str
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_payload(self) -> Dict:
        """Convert to Qdrant payload format."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "dominant_emotion": self.dominant_emotion,
            "valence": self.valence,
            "arousal": self.arousal,
            "confidence": self.confidence,
            "context": self.context,
        }
    
    @classmethod
    def from_payload(cls, id: str, payload: Dict) -> "EmotionalEpisode":
        """Create from Qdrant payload."""
        return cls(
            id=id,
            user_id=payload.get("user_id", ""),
            session_id=payload.get("session_id", ""),
            timestamp=datetime.fromisoformat(payload.get("timestamp", datetime.now().isoformat())),
            summary=payload.get("summary", ""),
            dominant_emotion=payload.get("dominant_emotion", "neutral"),
            valence=payload.get("valence", 0.0),
            arousal=payload.get("arousal", 0.5),
            confidence=payload.get("confidence", 0.0),
            context=payload.get("context", {}),
        )


class EpisodicMemoryManager:
    """
    Manages episodic memory storage and retrieval via Qdrant.
    
    Features:
    - Vector embedding of emotional experiences
    - Semantic similarity search
    - User-scoped queries
    - Privacy-preserving deletion
    """
    
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection: str = COLLECTION_NAME,
    ):
        """
        Initialize episodic memory manager.
        
        Args:
            qdrant_url: Qdrant server URL
            collection: Collection name for episodes
        """
        self.qdrant_url = qdrant_url
        self.collection = collection
        self._client = None
        self._embedder = None
        self._initialized = False
        
        logger.info(f"EpisodicMemoryManager created (qdrant: {qdrant_url})")
    
    async def initialize(self) -> bool:
        """
        Initialize Qdrant client and embedder.
        
        Returns:
            True if successful, False otherwise.
        """
        if self._initialized:
            return True
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            # Connect to Qdrant
            self._client = QdrantClient(url=self.qdrant_url)
            
            # Check/create collection
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection not in collection_names:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection}")
            
            # Load embedder
            self._load_embedder()
            
            self._initialized = True
            logger.info("Episodic memory initialized successfully")
            return True
            
        except ImportError:
            logger.error("qdrant-client not installed. Run: pip install qdrant-client")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize episodic memory: {e}")
            return False
    
    def _load_embedder(self) -> None:
        """Load sentence transformer for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence embedder loaded: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not installed, using fallback")
            self._embedder = None
    
    def _embed(self, text: str) -> List[float]:
        """Embed text to vector."""
        if self._embedder is None:
            # Fallback: random embedding (for testing only)
            import random
            return [random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]
        
        embedding = self._embedder.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    async def store_episode(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        emotional_state: Dict,
        context: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Store an emotional episode.
        
        Args:
            user_id: User identifier (will be hashed)
            session_id: Session UUID
            summary: Text summary of the moment
            emotional_state: Dict with dominant_emotion, valence, arousal
            context: Additional context (entities, topics)
        
        Returns:
            Episode ID if successful, None otherwise.
        """
        if not await self.initialize():
            return None
        
        try:
            from qdrant_client.models import PointStruct
            
            # Hash user ID for privacy
            user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
            
            # Create episode
            episode_id = str(uuid4())
            episode = EmotionalEpisode(
                id=episode_id,
                user_id=user_hash,
                session_id=str(session_id),
                timestamp=datetime.utcnow(),
                summary=summary,
                dominant_emotion=emotional_state.get("dominant_emotion", "neutral"),
                valence=emotional_state.get("valence", 0.0),
                arousal=emotional_state.get("arousal", 0.5),
                confidence=emotional_state.get("confidence", 0.0),
                context=context or {},
            )
            
            # Create embedding from summary + emotion
            embed_text = f"{summary} [Feeling: {episode.dominant_emotion}]"
            vector = self._embed(embed_text)
            
            # Store in Qdrant
            self._client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=episode_id,
                        vector=vector,
                        payload=episode.to_payload()
                    )
                ]
            )
            
            logger.info(f"Stored episode: {episode_id} for user {user_hash}")
            return episode_id
            
        except Exception as e:
            logger.error(f"Failed to store episode: {e}")
            return None
    
    async def find_similar(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        min_score: float = SIMILARITY_THRESHOLD,
    ) -> List[EmotionalEpisode]:
        """
        Find similar emotional episodes.
        
        Args:
            query: Search query (e.g., "times I felt lonely")
            user_id: User identifier
            limit: Max results to return
            min_score: Minimum similarity score
        
        Returns:
            List of similar episodes.
        """
        if not await self.initialize():
            return []
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Hash user ID
            user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
            
            # Embed query
            query_vector = self._embed(query)
            
            # Search with user filter
            results = self._client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_hash)
                        )
                    ]
                ),
                limit=limit,
                score_threshold=min_score,
            )
            
            # Convert to episodes
            episodes = []
            for result in results:
                episode = EmotionalEpisode.from_payload(
                    id=str(result.id),
                    payload=result.payload
                )
                episodes.append(episode)
            
            logger.info(f"Found {len(episodes)} similar episodes for query: {query[:50]}")
            return episodes
            
        except Exception as e:
            logger.error(f"Failed to search episodes: {e}")
            return []
    
    async def get_emotional_history(
        self,
        user_id: str,
        emotion: Optional[str] = None,
        limit: int = 10,
    ) -> List[EmotionalEpisode]:
        """
        Get user's emotional history.
        
        Args:
            user_id: User identifier
            emotion: Optional filter by emotion type
            limit: Max results
        
        Returns:
            List of episodes sorted by time.
        """
        if not await self.initialize():
            return []
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
            
            # Build filter
            must_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_hash))
            ]
            if emotion:
                must_conditions.append(
                    FieldCondition(key="dominant_emotion", match=MatchValue(value=emotion))
                )
            
            # Scroll through results
            results, _ = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=must_conditions),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            
            # Convert and sort by timestamp
            episodes = [
                EmotionalEpisode.from_payload(str(r.id), r.payload)
                for r in results
            ]
            episodes.sort(key=lambda e: e.timestamp, reverse=True)
            
            return episodes
            
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []
    
    async def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all episodes for a user (GDPR compliance).
        
        Args:
            user_id: User identifier
        
        Returns:
            True if successful.
        """
        if not await self.initialize():
            return False
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
            
            self._client.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_hash))
                    ]
                )
            )
            
            logger.info(f"Deleted all episodes for user: {user_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user data: {e}")
            return False
    
    async def get_context_for_llm(
        self,
        query: str,
        user_id: str,
        limit: int = 3,
    ) -> str:
        """
        Get formatted context for LLM prompt injection.
        
        Args:
            query: Current conversation context
            user_id: User identifier
            limit: Max episodes to include
        
        Returns:
            Formatted context string.
        """
        episodes = await self.find_similar(query, user_id, limit=limit)
        
        if not episodes:
            return ""
        
        lines = ["[CONTEXT: Similar past experiences]"]
        for ep in episodes:
            lines.append(f"- {ep.timestamp.strftime('%Y-%m-%d')}: {ep.summary}")
            lines.append(f"  (Feeling: {ep.dominant_emotion}, intensity: {ep.arousal:.1f})")
        
        return "\n".join(lines)


# Singleton
_manager: Optional[EpisodicMemoryManager] = None


def get_episodic_memory() -> EpisodicMemoryManager:
    """Get singleton episodic memory manager."""
    global _manager
    if _manager is None:
        _manager = EpisodicMemoryManager()
    return _manager
