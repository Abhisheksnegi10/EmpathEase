"""
Memory Consolidation - Background Processing ("Dreaming").

Runs asynchronously after session ends to:
1. Summarize session into key facts
2. Extract emotional peaks → Store in episodic memory (Qdrant)
3. Extract entities → Update semantic memory (Neo4j)
4. Generate user-facing insights

Uses LLM for summarization and extraction.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of memory consolidation."""
    session_id: str
    user_id: str
    summary: str
    emotional_peak: Optional[Dict]
    entities_extracted: List[str]
    insight: Optional[str]
    success: bool
    error: Optional[str] = None


class MemoryConsolidator:
    """
    Consolidates session memories after session ends.
    
    Pipeline:
    1. Load working memory (Redis)
    2. LLM summarization
    3. Store emotional peak (Qdrant)
    4. Update knowledge graph (Neo4j)
    5. Generate insight
    """
    
    def __init__(self):
        """Initialize consolidator."""
        self._llm = None
        self._episodic = None
        self._semantic = None
        logger.info("MemoryConsolidator initialized")
    
    async def consolidate(
        self,
        session_id: str,
        user_id: str,
        conversation_turns: List[Dict],
        emotional_states: List[Dict],
    ) -> ConsolidationResult:
        """
        Run full consolidation pipeline.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            conversation_turns: List of conversation turns
            emotional_states: List of emotional state snapshots
        
        Returns:
            ConsolidationResult with summary and extracted data.
        """
        try:
            # 0. Scrub PII from conversation before processing
            scrubbed_turns = await self._scrub_pii(conversation_turns)
            
            # 1. Generate session summary
            summary = await self._generate_summary(scrubbed_turns)
            
            # 2. Find emotional peak
            peak = self._find_emotional_peak(emotional_states)
            
            # 3. Store in episodic memory (with scrubbed summary)
            if peak:
                await self._store_episode(
                    user_id=user_id,
                    session_id=session_id,
                    summary=summary,
                    peak=peak,
                )
            
            # 4. Extract and store entities (from scrubbed text)
            entities = await self._extract_entities(user_id, scrubbed_turns)
            
            # 5. Generate insight
            insight = await self._generate_insight(summary, peak, entities)
            
            logger.info(f"Consolidation complete for session {session_id}")
            
            return ConsolidationResult(
                session_id=session_id,
                user_id=user_id,
                summary=summary,
                emotional_peak=peak,
                entities_extracted=entities,
                insight=insight,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Consolidation failed for session {session_id}: {e}")
            return ConsolidationResult(
                session_id=session_id,
                user_id=user_id,
                summary="",
                emotional_peak=None,
                entities_extracted=[],
                insight=None,
                success=False,
                error=str(e),
            )
    
    async def _scrub_pii(
        self,
        turns: List[Dict],
    ) -> List[Dict]:
        """Scrub PII from conversation turns before storage."""
        try:
            from app.services.privacy import get_scrubber
            
            scrubber = get_scrubber()
            scrubbed_turns = []
            
            for turn in turns:
                scrubbed_turn = turn.copy()
                
                # Scrub user input
                if "user_input" in turn:
                    scrubbed_turn["user_input"] = scrubber.scrub(
                        turn["user_input"], return_entities=False
                    ).scrubbed_text
                if "user_message" in turn:
                    scrubbed_turn["user_message"] = scrubber.scrub(
                        turn["user_message"], return_entities=False
                    ).scrubbed_text
                
                scrubbed_turns.append(scrubbed_turn)
            
            pii_count = sum(
                scrubber.scrub(t.get("user_input", ""), return_entities=False).pii_count
                for t in turns
            )
            if pii_count > 0:
                logger.info(f"Scrubbed {pii_count} PII items from conversation")
            
            return scrubbed_turns
            
        except Exception as e:
            logger.warning(f"PII scrubbing failed, using original: {e}")
            return turns
    
    async def _generate_summary(
        self,
        turns: List[Dict],
    ) -> str:
        """Generate session summary via simple extraction."""
        if not turns:
            return "No conversation recorded."
        
        # Simple extractive summary
        if turns:
            first_user = turns[0].get("user_input", turns[0].get("user_message", ""))[:200]
            return f"Session started with: {first_user}..."
        return "Session completed."
    
    def _find_emotional_peak(
        self,
        states: List[Dict],
    ) -> Optional[Dict]:
        """Find most intense emotional moment."""
        if not states:
            return None
        
        peak = None
        max_intensity = 0
        
        for state in states:
            # Calculate intensity as combination of arousal and abs(valence)
            arousal = state.get("arousal", 0.5)
            valence = state.get("valence", 0)
            intensity = arousal * 0.6 + abs(valence) * 0.4
            
            if intensity > max_intensity:
                max_intensity = intensity
                peak = state
        
        return peak
    
    async def _store_episode(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        peak: Dict,
    ) -> None:
        """Store emotional episode in Qdrant."""
        try:
            from app.memory.episodic import get_episodic_memory
            
            episodic = get_episodic_memory()
            await episodic.store_episode(
                user_id=user_id,
                session_id=session_id,
                summary=summary,
                emotional_state=peak,
            )
            
        except Exception as e:
            logger.warning(f"Failed to store episode: {e}")
    
    async def _extract_entities(
        self,
        user_id: str,
        turns: List[Dict],
    ) -> List[str]:
        """Extract and store entities from conversation."""
        entities = []
        
        try:
            from app.memory.semantic import get_semantic_memory
            
            semantic = get_semantic_memory()
            
            # Extract from each turn
            for turn in turns:
                text = turn.get("user_input", turn.get("user_message", ""))
                if text:
                    extracted = await semantic.extract_and_store_entities(user_id, text)
                    entities.extend(extracted)
            
            # Deduplicate
            entities = list(set(entities))
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
        
        return entities
    
    async def _generate_insight(
        self,
        summary: str,
        peak: Optional[Dict],
        entities: List[str],
    ) -> Optional[str]:
        """Generate a brief insight for the user."""
        if not peak:
            return None
        
        emotion = peak.get("dominant_emotion", "complex")
        valence = peak.get("valence", 0)
        
        # Simple insights based on patterns
        if valence < -0.5:
            return f"You experienced some difficult emotions today. It's okay to feel {emotion}."
        elif valence > 0.5:
            return f"You showed strength and resilience in today's session."
        elif entities:
            return f"You mentioned {entities[0]} - this seems to be important to you."
        else:
            return "Thank you for sharing today. Each conversation is a step forward."


# Background task function (for Celery)
async def consolidate_session_task(
    session_id: str,
    user_id: str,
) -> ConsolidationResult:
    """
    Background task to consolidate session memory.
    
    Called after session ends via Celery.
    """
    from app.memory.working import WorkingMemoryManager
    from uuid import UUID
    
    # Load working memory
    wm = WorkingMemoryManager(UUID(session_id))
    state = await wm.get_state()
    
    if not state:
        logger.warning(f"No working memory found for session {session_id}")
        return ConsolidationResult(
            session_id=session_id,
            user_id=user_id,
            summary="",
            emotional_peak=None,
            entities_extracted=[],
            insight=None,
            success=False,
            error="No working memory found",
        )
    
    # Run consolidation
    consolidator = MemoryConsolidator()
    result = await consolidator.consolidate(
        session_id=session_id,
        user_id=user_id,
        conversation_turns=[t.model_dump() for t in state.turns],
        emotional_states=[t.emotional_state.model_dump() for t in state.turns],
    )
    
    # Cleanup working memory after successful consolidation
    if result.success:
        await wm.cleanup()
    
    return result


# Singleton
_consolidator: Optional[MemoryConsolidator] = None


def get_consolidator() -> MemoryConsolidator:
    """Get singleton memory consolidator."""
    global _consolidator
    if _consolidator is None:
        _consolidator = MemoryConsolidator()
    return _consolidator
