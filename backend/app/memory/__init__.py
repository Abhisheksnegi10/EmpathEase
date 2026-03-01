"""
Memory systems module.

Components:
- working: Redis-based session state (short-term)
- episodic: Qdrant vector store for emotional moments (long-term)
- semantic: Neo4j knowledge graph for relationships (long-term)
- consolidation: Background processing after sessions
"""

from app.memory.working import WorkingMemoryManager, get_working_memory
from app.memory.episodic import (
    EpisodicMemoryManager,
    EmotionalEpisode,
    get_episodic_memory,
)
from app.memory.semantic import (
    SemanticMemoryManager,
    Entity,
    get_semantic_memory,
)
from app.memory.consolidation import (
    MemoryConsolidator,
    ConsolidationResult,
    get_consolidator,
    consolidate_session_task,
)

__all__ = [
    # Working memory
    'WorkingMemoryManager',
    'get_working_memory',
    # Episodic memory
    'EpisodicMemoryManager',
    'EmotionalEpisode',
    'get_episodic_memory',
    # Semantic memory
    'SemanticMemoryManager',
    'Entity',
    'get_semantic_memory',
    # Consolidation
    'MemoryConsolidator',
    'ConsolidationResult',
    'get_consolidator',
    'consolidate_session_task',
]
