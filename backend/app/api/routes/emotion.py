"""
Emotion Analysis API Routes.

Provides endpoints for:
- Text emotion prediction (multilingual)
- Facial emotion analysis
"""

import base64
from typing import List, Optional
from io import BytesIO

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

from app.ml import predict_text_emotion, get_text_emotion_classifier, EMOTION_LABELS
from app.ml.facial_affect import predict_emotion as predict_facial_emotion

router = APIRouter(prefix="/emotion", tags=["emotion"])


# ============================================================
# Request/Response Schemas
# ============================================================

class TextEmotionRequest(BaseModel):
    """Request for text emotion analysis."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    language: Optional[str] = Field(None, description="Optional language hint (auto-detected if not provided)")


class EmotionScores(BaseModel):
    """Emotion probability scores."""
    anger: float = Field(..., ge=0, le=1)
    disgust: float = Field(..., ge=0, le=1)
    fear: float = Field(..., ge=0, le=1)
    joy: float = Field(..., ge=0, le=1)
    sadness: float = Field(..., ge=0, le=1)
    surprise: float = Field(..., ge=0, le=1)
    neutral: float = Field(..., ge=0, le=1)


class TextEmotionResponse(BaseModel):
    """Response for text emotion analysis."""
    success: bool = True
    text: str
    language: str
    dominant_emotion: str
    confidence: float
    emotions: EmotionScores
    




class BatchTextRequest(BaseModel):
    """Request for batch text analysis."""
    texts: List[str] = Field(..., min_items=1, max_items=20)


class BatchEmotionResult(BaseModel):
    """Single result in batch response."""
    text: str
    language: str
    dominant_emotion: str
    confidence: float


class BatchTextResponse(BaseModel):
    """Response for batch text analysis."""
    success: bool = True
    count: int
    results: List[BatchEmotionResult]


# ============================================================
# API Endpoints
# ============================================================

@router.post("/analyze", response_model=TextEmotionResponse)
async def analyze_text_emotion(request: TextEmotionRequest):
    """
    Analyze emotion in text.
    
    Supports English, Hindi, and Hinglish (code-mixed) text.
    Language is auto-detected if not provided.
    
    Returns:
        Dominant emotion, confidence, and all emotion probabilities
    """
    try:
        language = request.language or 'en'
        
        # Get emotion prediction
        result = predict_text_emotion(request.text, language)
        
        return TextEmotionResponse(
            success=True,
            text=request.text,
            language=result.language or language,
            dominant_emotion=result.dominant_emotion,
            confidence=result.confidence,
            emotions=EmotionScores(**result.emotions)
        )
        
    except Exception as e:
        import traceback
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Emotion analysis error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Emotion analysis failed: {str(e)}")





@router.post("/analyze-batch", response_model=BatchTextResponse)
async def analyze_text_batch(request: BatchTextRequest):
    """
    Analyze emotions in multiple texts (batch processing).
    
    More efficient than calling /analyze multiple times.
    Maximum 20 texts per request.
    """
    try:
        classifier = get_text_emotion_classifier()
        results = classifier.predict_batch(request.texts)
        
        batch_results = [
            BatchEmotionResult(
                text=r.text[:100],  # Truncate for response
                language=r.language or 'en',
                dominant_emotion=r.dominant_emotion,
                confidence=r.confidence
            )
            for r in results
        ]
        
        return BatchTextResponse(
            success=True,
            count=len(batch_results),
            results=batch_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get("/labels")
async def get_emotion_labels():
    """Get list of supported emotion labels."""
    return {
        "labels": EMOTION_LABELS,
        "count": len(EMOTION_LABELS)
    }


@router.get("/health")
async def health_check():
    """Check if emotion analysis service is ready."""
    try:
        classifier = get_text_emotion_classifier()
        # Test with a simple phrase
        result = classifier.predict("test")
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": str(classifier.device),
            "labels": len(EMOTION_LABELS)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }


# ============================================================
# Facial Emotion Analysis
# ============================================================

class FaceAnalysisRequest(BaseModel):
    """Request for facial emotion analysis."""
    image_base64: str = Field(..., description="Base64-encoded image (JPEG/PNG)")


class FaceAnalysisResponse(BaseModel):
    """Response for facial emotion analysis."""
    success: bool = True
    top_emotion: str = Field(..., description="Dominant emotion detected")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    emotions: dict = Field(..., description="All emotion probabilities")


@router.post("/analyze-face", response_model=FaceAnalysisResponse)
async def analyze_face_emotion(request: FaceAnalysisRequest):
    """
    Analyze emotion from a face image.
    
    Accepts a base64-encoded image (JPEG or PNG) from webcam or file.
    Returns detected emotion with confidence scores.
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Convert to numpy array for the model
        image_array = np.array(image)
        
        # Predict emotion
        result = predict_facial_emotion(image_array)
        
        return FaceAnalysisResponse(
            success=True,
            top_emotion=result['top_emotion'],
            confidence=result['confidence'],
            emotions=result['all_probs']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face analysis failed: {str(e)}")


# ============================================================
# Multimodal Fusion Analysis
# ============================================================

from app.schemas.emotion import MultimodalRequest, MultimodalResponse
from app.ml.fusion import fuse_emotions


@router.post("/analyze-multimodal", response_model=MultimodalResponse)
async def analyze_multimodal(request: MultimodalRequest):
    """
    Fused multimodal emotion analysis.

    Accepts any combination of text, face image, and audio.
    Weights are driven by STT confidence (cascade tiers):
      - High  (≥0.85): text=0.60, vocal=0.30, face=0.10
      - Medium (0.65–0.85): text=0.40, vocal=0.30, face=0.30
      - Low   (<0.65): text=0.00, vocal=0.60, face=0.40
    """
    import time as _time
    start = _time.perf_counter()

    try:
        text_result = None
        face_result = None
        vocal_result = None

        # --- Text modality ---
        if request.text:
            text_result = predict_text_emotion(
                request.text, request.language
            )

        # --- Face modality ---
        if request.image_base64:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            image_array = np.array(image)
            face_result = predict_facial_emotion(image_array)

        # --- Vocal modality ---
        if request.audio_base64:
            from app.ml.vocal_prosody import predict_vocal_emotion
            import soundfile as sf

            audio_bytes = base64.b64decode(request.audio_base64)
            audio_buf = BytesIO(audio_bytes)

            try:
                audio, sr = sf.read(audio_buf)
            except Exception:
                # Fallback: try raw float32 PCM
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
                sr = request.audio_sample_rate

            vocal_result = predict_vocal_emotion(audio, sr)

        # --- Fuse ---
        fused_state = fuse_emotions(
            text_result=text_result,
            face_result=face_result,
            vocal_result=vocal_result,
            stt_confidence=request.stt_confidence,
        )

        elapsed_ms = (_time.perf_counter() - start) * 1000

        return MultimodalResponse(
            success=True,
            fused_state=fused_state,
            inference_time_ms=round(elapsed_ms, 1),
        )

    except Exception as e:
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"Multimodal analysis error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Multimodal analysis failed: {str(e)}",
        )
