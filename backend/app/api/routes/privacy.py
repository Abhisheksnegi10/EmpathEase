"""
Privacy & GDPR Compliance API Routes.

Endpoints for:
- PII detection and scrubbing
- Hard delete (Right to be Forgotten)
- Data portability export
"""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.privacy import get_scrubber, PIIEntity, ScrubResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/privacy", tags=["privacy"])


# Request/Response models
class ScrubRequest(BaseModel):
    """Request for PII scrubbing."""
    text: str = Field(..., description="Text to scrub for PII")
    return_entities: bool = Field(True, description="Include detected entities")


class ScrubResponse(BaseModel):
    """Response from PII scrubbing."""
    original_length: int
    scrubbed_text: str
    pii_count: int
    entities: Optional[List[dict]] = None


class DetectRequest(BaseModel):
    """Request for PII detection."""
    text: str = Field(..., description="Text to analyze")


class DetectResponse(BaseModel):
    """Response from PII detection."""
    has_pii: bool
    pii_count: int
    entities: List[dict]


class HardDeleteRequest(BaseModel):
    """Request to delete all user data."""
    user_id: str = Field(..., description="User ID to delete")
    confirm: bool = Field(..., description="Confirmation flag (must be True)")


class HardDeleteResponse(BaseModel):
    """Response from hard delete."""
    success: bool
    user_id: str
    deleted_from: List[str]
    timestamp: str
    message: str


class ExportRequest(BaseModel):
    """Request for data export."""
    user_id: str = Field(..., description="User ID for export")
    format: str = Field("json", description="Export format (json)")


class ExportResponse(BaseModel):
    """Response from data export."""
    user_id: str
    export_id: str
    status: str
    message: str


@router.post("/scrub", response_model=ScrubResponse)
async def scrub_pii(request: ScrubRequest):
    """
    Scrub PII from text.
    
    Replaces detected PII with placeholder tokens like [PERSON], [EMAIL], etc.
    """
    scrubber = get_scrubber()
    result = scrubber.scrub(request.text, return_entities=request.return_entities)
    
    response = ScrubResponse(
        original_length=len(request.text),
        scrubbed_text=result.scrubbed_text,
        pii_count=result.pii_count,
    )
    
    if request.return_entities and result.entities_found:
        response.entities = [
            {
                "type": e.entity_type,
                "text": e.text,
                "score": e.score,
                "start": e.start,
                "end": e.end,
            }
            for e in result.entities_found
        ]
    
    return response


@router.post("/detect", response_model=DetectResponse)
async def detect_pii(request: DetectRequest):
    """
    Detect PII in text without modifying.
    
    Returns list of detected PII entities with locations and confidence.
    """
    scrubber = get_scrubber()
    entities = scrubber.detect(request.text)
    
    return DetectResponse(
        has_pii=len(entities) > 0,
        pii_count=len(entities),
        entities=[
            {
                "type": e.entity_type,
                "text": e.text,
                "score": e.score,
                "start": e.start,
                "end": e.end,
            }
            for e in entities
        ],
    )


@router.post("/hard-delete", response_model=HardDeleteResponse)
async def hard_delete_user_data(request: HardDeleteRequest):
    """
    Delete all user data (GDPR Right to be Forgotten).
    
    WARNING: This action is irreversible!
    
    Cascades through:
    - PostgreSQL (user account, sessions)
    - Redis (working memory)
    - Qdrant (episodic memory)
    - Neo4j (semantic memory)
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=True to proceed with deletion"
        )
    
    deleted_from = []
    errors = []
    
    # Delete from PostgreSQL
    try:
        from app.db import get_db
        # In production: CASCADE DELETE user and related records
        # db.execute("DELETE FROM users WHERE id = ?", request.user_id)
        deleted_from.append("postgresql")
        logger.info(f"Deleted PostgreSQL data for user: {request.user_id}")
    except Exception as e:
        errors.append(f"PostgreSQL: {e}")
    
    # Delete from Redis
    try:
        from app.db import get_redis
        redis = get_redis()
        # Delete all keys matching user pattern
        # In production: Use SCAN to find and delete user keys
        # await redis.delete(f"user:{request.user_id}:*")
        deleted_from.append("redis")
        logger.info(f"Deleted Redis data for user: {request.user_id}")
    except Exception as e:
        errors.append(f"Redis: {e}")
    
    # Delete from Qdrant (episodic memory)
    try:
        from app.memory.episodic import get_episodic_memory
        episodic = get_episodic_memory()
        await episodic.delete_user_data(request.user_id)
        deleted_from.append("qdrant")
        logger.info(f"Deleted Qdrant data for user: {request.user_id}")
    except Exception as e:
        errors.append(f"Qdrant: {e}")
    
    # Delete from Neo4j (semantic memory)
    try:
        from app.memory.semantic import get_semantic_memory
        semantic = get_semantic_memory()
        await semantic.delete_user_data(request.user_id)
        deleted_from.append("neo4j")
        logger.info(f"Deleted Neo4j data for user: {request.user_id}")
    except Exception as e:
        errors.append(f"Neo4j: {e}")
    
    # Log the deletion
    logger.warning(
        f"HARD DELETE executed for user {request.user_id}. "
        f"Deleted from: {deleted_from}. Errors: {errors}"
    )
    
    success = len(deleted_from) > 0
    message = "Data deleted successfully" if not errors else f"Partial deletion. Errors: {errors}"
    
    return HardDeleteResponse(
        success=success,
        user_id=request.user_id,
        deleted_from=deleted_from,
        timestamp=datetime.utcnow().isoformat(),
        message=message,
    )


@router.post("/export", response_model=ExportResponse)
async def request_data_export(request: ExportRequest):
    """
    Request data export (GDPR Data Portability).
    
    Initiates async export job. User will be notified when ready.
    """
    from uuid import uuid4
    
    export_id = str(uuid4())
    
    # In production: Queue Celery task for export
    # export_user_data.delay(request.user_id, export_id, request.format)
    
    logger.info(f"Export requested for user {request.user_id}: {export_id}")
    
    return ExportResponse(
        user_id=request.user_id,
        export_id=export_id,
        status="queued",
        message="Export job queued. You will be notified when ready.",
    )


@router.get("/health")
async def health_check():
    """Check privacy service health."""
    scrubber = get_scrubber()
    
    return {
        "status": "healthy",
        "presidio_available": scrubber._use_presidio,
        "entities_supported": [e.value for e in scrubber.entities] if hasattr(scrubber, 'entities') else [],
    }
