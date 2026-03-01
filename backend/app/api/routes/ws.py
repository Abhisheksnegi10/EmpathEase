"""
WebSocket Route — per-message processing loop.

Pipeline per turn:
    1. Sarvam STT (if audio) → transcript + stt_confidence
    2. Parallel emotion analysis: FER + text + vocal
    3. Fusion → FusedEmotionalState
    4. Crisis check → if urgent: bypass LLM, return template
    5. Therapy engine → LLM response (full, no streaming)
    6. Sarvam TTS (if voice session) → audio response
    7. Memory update → working.py stores turn
    8. Send response to client
"""

import base64
import json
import logging
import time
from io import BytesIO
from typing import Dict, List, Optional
from uuid import UUID, uuid4

import httpx
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.core.websocket_manager import ConnectionManager
from app.schemas.emotion import FusedEmotionalState
from app.ml.fusion import fuse_emotions
from app.services.crisis import assess_crisis
from app.services.therapy import get_therapy_engine
from app.memory.working import get_working_memory

logger = logging.getLogger(__name__)
router = APIRouter()
manager = ConnectionManager()
settings = get_settings()


# ============================================================================
# WebSocket endpoint
# ============================================================================

@router.websocket("/ws/session/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str):
    """
    Main therapy session WebSocket.

    Expects JSON messages:
        {
            "text": "optional user text",
            "audio_base64": "optional base64 audio",
            "image_base64": "optional base64 face image",
            "is_voice_session": false
        }

    Sends JSON responses:
        {
            "type": "response",
            "text": "therapy response",
            "audio_base64": "optional TTS audio",
            "fused_state": { ... },
            "crisis_level": "none"
        }
    """
    # Accept any session ID format — derive a UUID if it's not one already
    try:
        sid = UUID(session_id)
    except (ValueError, AttributeError):
        from uuid import NAMESPACE_URL, uuid5
        sid = uuid5(NAMESPACE_URL, session_id)
    user_id = f"user_{sid}"

    # Connect
    await manager.connect(websocket, sid, user_id)
    memory = get_working_memory(sid)
    await memory.initialize()
    therapy = get_therapy_engine()
    conversation_history: List[Dict[str, str]] = []

    logger.info("Session started: %s", sid)

    try:
        while True:
            # Receive message
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error(websocket, "Invalid JSON")
                continue

            start = time.perf_counter()
            is_voice = msg.get("is_voice_session", False)

            try:
                # === Step 1: STT (if audio) ===
                user_text = msg.get("text", "")
                stt_confidence = None
                audio_array = None
                audio_sr = 16000

                if msg.get("audio_base64"):
                    stt_result = await _sarvam_stt(msg["audio_base64"])
                    if stt_result:
                        user_text = stt_result["transcript"]
                        stt_confidence = stt_result["confidence"]

                    # Decode audio for vocal emotion
                    try:
                        audio_bytes = base64.b64decode(msg["audio_base64"])
                        import soundfile as sf
                        audio_buf = BytesIO(audio_bytes)
                        audio_array, audio_sr = sf.read(audio_buf)
                    except Exception as e:
                        logger.warning("Audio decode failed: %s", e)

                if not user_text.strip():
                    # Image-only frame → run FER and send lightweight state update
                    if msg.get("image_base64"):
                        try:
                            from app.ml.facial_affect import predict_emotion as predict_facial_emotion
                            from PIL import Image
                            img_bytes = base64.b64decode(msg["image_base64"])
                            image = Image.open(BytesIO(img_bytes)).convert("RGB")
                            face_result = predict_facial_emotion(np.array(image))
                            fused = fuse_emotions(face_result=face_result)
                            await websocket.send_json({
                                "type": "state_update",
                                "fused_state": fused.model_dump(),
                            })
                        except Exception as e:
                            logger.warning("FER-only frame failed: %s", e)
                        continue

                    await _send_error(websocket, "No text or transcribable audio received")
                    continue

                # === Step 2: Emotion analysis ===
                text_result = None
                face_result = None
                vocal_result = None

                # Text emotion
                try:
                    from app.ml.text_emotion import predict_text_emotion
                    text_result = predict_text_emotion(user_text)
                except Exception as e:
                    logger.warning("Text emotion failed: %s", e)

                # Facial emotion
                if msg.get("image_base64"):
                    try:
                        from app.ml.facial_affect import predict_emotion as predict_facial_emotion
                        from PIL import Image
                        img_bytes = base64.b64decode(msg["image_base64"])
                        image = Image.open(BytesIO(img_bytes)).convert("RGB")
                        face_result = predict_facial_emotion(np.array(image))
                    except Exception as e:
                        logger.warning("Facial emotion failed: %s", e)

                # Vocal emotion
                if audio_array is not None:
                    try:
                        from app.ml.vocal_prosody import predict_vocal_emotion
                        vocal_result = predict_vocal_emotion(audio_array, audio_sr)
                    except Exception as e:
                        logger.warning("Vocal emotion failed: %s", e)

                # === Step 3: Fusion ===
                fused_state = fuse_emotions(
                    text_result=text_result,
                    face_result=face_result,
                    vocal_result=vocal_result,
                    stt_confidence=stt_confidence,
                )

                # === Step 4: Crisis check ===
                crisis = assess_crisis(user_text, fused_state)
                fused_state.crisis_level = crisis.level

                if crisis.level == "urgent":
                    # Bypass LLM — send hard-coded crisis response
                    response_text = crisis.template_response
                    conversation_history.append({"role": "user", "content": user_text})
                    conversation_history.append({"role": "assistant", "content": response_text})
                else:
                    # === Step 5: Therapy engine ===
                    if crisis.level == "moderate":
                        # Inject crisis awareness but still call LLM
                        response_text = crisis.template_response
                    else:
                        response_text = await therapy.generate_response(
                            user_text=user_text,
                            fused_state=fused_state,
                            conversation_history=conversation_history,
                        )

                    conversation_history.append({"role": "user", "content": user_text})
                    conversation_history.append({"role": "assistant", "content": response_text})

                # Cap history
                max_turns = settings.max_conversation_turns * 2  # user + assistant per turn
                if len(conversation_history) > max_turns:
                    conversation_history = conversation_history[-max_turns:]

                # === Step 6: TTS (if voice session) ===
                audio_response = None
                if is_voice and response_text:
                    audio_response = await _sarvam_tts(response_text)

                # === Step 7: Memory update ===
                try:
                    await memory.add_turn(
                        user_input=user_text,
                        system_response=response_text,
                        emotional_state=fused_state.model_dump(),
                    )
                except Exception as e:
                    logger.warning("Memory update failed: %s", e)

                # === Step 8: Send response ===
                elapsed_ms = (time.perf_counter() - start) * 1000

                response_payload = {
                    "type": "response",
                    "text": response_text,
                    "user_text": user_text,          # transcription for voice messages
                    "fused_state": fused_state.model_dump(),
                    "crisis_level": crisis.level,
                    "inference_time_ms": round(elapsed_ms, 1),
                }
                if audio_response:
                    response_payload["audio_base64"] = audio_response

                await websocket.send_json(response_payload)

            except Exception as e:
                logger.error("Turn processing error: %s", e, exc_info=True)
                await _send_error(websocket, f"Processing error: {str(e)}")

    except WebSocketDisconnect:
        logger.info("Session disconnected: %s", sid)
    except Exception as e:
        logger.error("Session error: %s", e, exc_info=True)
    finally:
        # Flush to episodic memory on disconnect
        try:
            state = await memory.cleanup()
            if state and state.turns:
                logger.info(
                    "Session %s: flushed %d turns to episodic memory",
                    sid, len(state.turns),
                )
                # TODO: episodic.store_session(sid, state)
        except Exception as e:
            logger.warning("Episodic flush failed: %s", e)

        manager.disconnect(sid, user_id)


# ============================================================================
# Sarvam STT
# ============================================================================

async def _sarvam_stt(audio_base64: str) -> Optional[Dict]:
    """
    Call Sarvam STT API to transcribe audio.

    Sarvam expects multipart/form-data with the audio as a file field.
    Returns:
        {"transcript": str, "confidence": float} or None
    """
    if not settings.sarvam_api_key:
        logger.warning("Sarvam API key not configured — skipping STT")
        return None

    try:
        # Decode base64 → raw audio bytes for multipart upload
        audio_bytes = base64.b64decode(audio_base64)

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                settings.sarvam_stt_url,
                headers={
                    "api-subscription-key": settings.sarvam_api_key,
                    # Content-Type set automatically by httpx for multipart
                },
                files={
                    "file": ("audio.webm", audio_bytes, "audio/webm"),
                },
                data={
                    "model": settings.sarvam_stt_model,
                },
            )
            response.raise_for_status()
            data = response.json()

            transcript = data.get("transcript", "")
            confidence = data.get("confidence", 0.5)

            logger.info(
                "STT: '%s' (confidence=%.2f)",
                transcript[:50], confidence,
            )
            return {"transcript": transcript, "confidence": confidence}

    except Exception as e:
        logger.error("Sarvam STT error: %s", e)
        return None


# ============================================================================
# Sarvam TTS
# ============================================================================

async def _sarvam_tts(text: str) -> Optional[str]:
    """
    Call Sarvam TTS API to synthesize speech.

    Returns:
        Base64-encoded audio string, or None
    """
    if not settings.sarvam_api_key:
        logger.warning("Sarvam API key not configured — skipping TTS")
        return None

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                settings.sarvam_tts_url,
                headers={
                    "api-subscription-key": settings.sarvam_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": [text[:500]],  # TTS limit; Sarvam expects a list
                    "target_language_code": "hi-IN",
                    "model": settings.sarvam_tts_model,
                    "speaker": settings.sarvam_tts_speaker,
                    "enable_preprocessing": True,  # cleans LLM markdown before TTS
                },
            )
            response.raise_for_status()
            data = response.json()
            # Sarvam TTS returns: {"audios": ["<base64_wav>", ...]}
            audios = data.get("audios", [])
            audio_b64 = audios[0] if audios else ""
            if audio_b64:
                logger.info("TTS: generated %d bytes", len(audio_b64))
            return audio_b64 or None

    except Exception as e:
        logger.error("Sarvam TTS error: %s", e)
        return None


# ============================================================================
# Helper
# ============================================================================

async def _send_error(websocket: WebSocket, detail: str):
    """Send error message to client."""
    await websocket.send_json({
        "type": "error",
        "detail": detail,
    })
