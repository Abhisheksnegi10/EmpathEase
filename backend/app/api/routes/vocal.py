"""
Vocal Prosody API Routes.

Endpoints for voice emotion analysis.
"""

import logging
from typing import List, Optional
from io import BytesIO

from fastapi import (
    APIRouter, 
    File, 
    UploadFile, 
    HTTPException, 
    Query,
    Body
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vocal", tags=["vocal"])


# Request/Response models
class VocalEmotionResponse(BaseModel):
    """Response for vocal emotion analysis."""
    dominant_emotion: str = Field(..., description="Most likely emotion")
    confidence: float = Field(..., description="Confidence score (0-1)")
    emotions: dict = Field(..., description="All emotion probabilities")
    valence: float = Field(..., description="Emotional valence (-1 to 1)")
    arousal: float = Field(..., description="Emotional arousal (0 to 1)")
    duration: float = Field(..., description="Audio duration in seconds")


class AudioChunkRequest(BaseModel):
    """Request for audio chunk analysis."""
    audio_base64: str = Field(..., description="Base64-encoded audio data")
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    format: str = Field("wav", description="Audio format (wav, mp3, etc)")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_id: str


@router.post("/analyze", response_model=VocalEmotionResponse)
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file to analyze"),
    sample_rate: int = Query(16000, description="Sample rate if known")
):
    """
    Analyze uploaded audio file for vocal emotion.
    
    Accepts common audio formats: WAV, MP3, OGG, FLAC.
    Audio is automatically resampled to 16kHz if needed.
    
    Returns emotion classification with confidence scores.
    """
    try:
        from app.ml.vocal_prosody import predict_vocal_emotion
        import numpy as np
        
        # Read file content
        content = await file.read()
        logger.info(f"Received audio file: {file.filename}, size: {len(content)} bytes")
        
        # Load audio based on format
        audio, sr = _load_audio_from_bytes(content, file.filename)
        logger.info(f"Audio loaded: samples={len(audio)}, sr={sr}, dtype={audio.dtype}")
        
        # Validate audio
        if len(audio) < 1000:
            raise ValueError(f"Audio too short: {len(audio)} samples (need at least 1000)")
        
        # Predict emotion
        logger.info("Running vocal emotion prediction...")
        result = predict_vocal_emotion(audio, sr)
        logger.info(f"Prediction result: {result.dominant_emotion} ({result.confidence:.2f})")
        
        return VocalEmotionResponse(
            dominant_emotion=result.dominant_emotion,
            confidence=result.confidence,
            emotions=result.emotions,
            valence=result.valence,
            arousal=result.arousal,
            duration=result.duration
        )
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing dependency missing: {e}"
        )
    except Exception as e:
        import traceback
        logger.error(f"Audio analysis error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze audio: {str(e)}"
        )


@router.post("/analyze-chunk", response_model=VocalEmotionResponse)
async def analyze_audio_chunk(
    request: AudioChunkRequest
):
    """
    Analyze a base64-encoded audio chunk.
    
    For streaming/real-time analysis. Send chunks of 200-500ms for best results.
    """
    try:
        from app.ml.vocal_prosody import predict_vocal_emotion
        import numpy as np
        import base64
        
        # Decode base64
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Load audio
        audio, sr = _load_audio_from_bytes(audio_bytes, f"chunk.{request.format}")
        
        # Predict emotion
        result = predict_vocal_emotion(audio, sr)
        
        return VocalEmotionResponse(
            dominant_emotion=result.dominant_emotion,
            confidence=result.confidence,
            emotions=result.emotions,
            valence=result.valence,
            arousal=result.arousal,
            duration=result.duration
        )
        
    except Exception as e:
        logger.error(f"Chunk analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze audio chunk: {str(e)}"
        )


@router.get("/labels")
async def get_emotion_labels() -> List[str]:
    """Get list of supported emotion labels."""
    from app.ml.vocal_prosody import EMOTION_LABELS
    return EMOTION_LABELS


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check for vocal analysis service.
    
    Returns model loading status and configuration.
    """
    from app.ml.vocal_prosody import get_vocal_analyzer, EMOTION2VEC_MODEL
    
    analyzer = get_vocal_analyzer()
    
    return HealthResponse(
        status="healthy" if analyzer.is_loaded() else "ready",
        model_loaded=analyzer.is_loaded(),
        model_id=EMOTION2VEC_MODEL
    )


def _load_audio_from_bytes(content: bytes, filename: str):
    """Load audio from bytes using available libraries."""
    import numpy as np
    import struct
    
    logger.info(f"Loading audio: {filename}, size: {len(content)} bytes")
    
    # For WAV files, try a direct WAV parser first (no ffmpeg needed)
    if filename.lower().endswith('.wav'):
        try:
            # Simple WAV parser for PCM format
            import io
            buf = io.BytesIO(content)
            
            # Read RIFF header
            riff = buf.read(4)
            if riff == b'RIFF':
                file_size = struct.unpack('<I', buf.read(4))[0]
                wave = buf.read(4)
                if wave == b'WAVE':
                    # Find fmt chunk
                    while True:
                        chunk_id = buf.read(4)
                        if len(chunk_id) < 4:
                            break
                        chunk_size = struct.unpack('<I', buf.read(4))[0]
                        
                        if chunk_id == b'fmt ':
                            fmt_data = buf.read(chunk_size)
                            audio_format = struct.unpack('<H', fmt_data[0:2])[0]
                            num_channels = struct.unpack('<H', fmt_data[2:4])[0]
                            sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                            bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
                        elif chunk_id == b'data':
                            audio_data = buf.read(chunk_size)
                            
                            # Convert to numpy array
                            if bits_per_sample == 16:
                                samples = np.frombuffer(audio_data, dtype=np.int16)
                                samples = samples.astype(np.float32) / 32768.0
                            elif bits_per_sample == 32:
                                samples = np.frombuffer(audio_data, dtype=np.int32)
                                samples = samples.astype(np.float32) / 2147483648.0
                            else:
                                samples = np.frombuffer(audio_data, dtype=np.uint8)
                                samples = (samples.astype(np.float32) - 128) / 128.0
                            
                            # Convert stereo to mono
                            if num_channels == 2:
                                samples = samples.reshape(-1, 2).mean(axis=1)
                            
                            logger.info(f"WAV loaded: {sample_rate}Hz, {len(samples)} samples, {num_channels}ch")
                            return samples, sample_rate
                        else:
                            buf.read(chunk_size)  # Skip unknown chunk
            
        except Exception as e:
            logger.warning(f"Direct WAV parsing failed: {e}")
    
    # Try soundfile (works with WAV, FLAC, OGG)
    try:
        import soundfile as sf
        audio, sr = sf.read(BytesIO(content))
        logger.info(f"Soundfile loaded: {sr}Hz, shape: {audio.shape}")
        return audio, sr
    except ImportError:
        logger.debug("soundfile not available")
    except Exception as e:
        logger.warning(f"Soundfile failed: {e}")
    
    # Try librosa (needs ffmpeg for some formats)
    try:
        import librosa
        audio, sr = librosa.load(BytesIO(content), sr=None)
        logger.info(f"Librosa loaded: {sr}Hz, shape: {audio.shape}")
        return audio, sr
    except ImportError:
        logger.debug("librosa not available")
    except Exception as e:
        logger.warning(f"Librosa failed: {e}")
    
    # Try scipy.io.wavfile for WAV
    if filename.lower().endswith('.wav'):
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(BytesIO(content))
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            logger.info(f"Scipy loaded: {sr}Hz, shape: {audio.shape}")
            return audio, sr
        except ImportError:
            logger.debug("scipy not available")
        except Exception as e:
            logger.warning(f"Scipy failed: {e}")
    
    raise ValueError(
        f"Could not load audio file {filename}. Make sure soundfile or scipy is installed."
    )

