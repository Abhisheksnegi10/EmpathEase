"""
Therapy Engine — LLM orchestration layer.

Owns:
- System prompt loading + per-turn EMOTIONAL_STATE injection (Option A)
- Conversation history management (in-memory list)
- Groq API (primary) → Ollama (fallback) switching
- Full response (no streaming)

Usage:
    from app.services.therapy import get_therapy_engine

    engine = get_therapy_engine()
    response = await engine.generate_response(
        user_text="I feel so alone",
        fused_state=fused_state,
        conversation_history=history,
    )
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from app.config import get_settings
from app.schemas.emotion import FusedEmotionalState

logger = logging.getLogger(__name__)

# System prompt template path
_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"

# Fallback response when both LLMs fail
_FALLBACK_RESPONSE = (
    "Main samajh sakta/sakti hoon ki aap kuch important share kar rahe hain. "
    "Abhi mujhe thoda technical issue aa raha hai, lekin main yahan hoon. "
    "Kya aap mujhe dobara bata sakte hain?"
)


class TherapyEngine:
    """
    LLM-backed therapy response generator.

    Groq primary → Ollama fallback.
    Option A: rebuilds system prompt each turn (stateless, debuggable).
    """

    def __init__(self):
        self._settings = get_settings()
        self._prompt_template = self._load_prompt_template()
        self._http_client = httpx.AsyncClient(timeout=30.0)

        logger.info(
            "TherapyEngine initialized | groq=%s | ollama=%s",
            self._settings.groq_model,
            f"{self._settings.ollama_host}/{self._settings.ollama_model}",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_response(
        self,
        user_text: str,
        fused_state: FusedEmotionalState,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a therapy response for the user's message.

        Args:
            user_text: The user's message
            fused_state: Current fused emotional state
            conversation_history: List of {"role": "user"|"assistant", "content": ...}

        Returns:
            Complete response text
        """
        # Build system prompt with injected emotional state
        system_prompt = self._build_system_prompt(fused_state)

        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        # Try Groq first, fall back to Ollama
        response = await self._call_groq(messages)
        if response is None:
            logger.warning("Groq failed, falling back to Ollama")
            response = await self._call_ollama(messages)
        if response is None:
            logger.error("Both Groq and Ollama failed — using fallback template")
            response = _FALLBACK_RESPONSE

        return response

    async def close(self):
        """Close the HTTP client."""
        await self._http_client.aclose()

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _load_prompt_template(self) -> str:
        """Load system prompt template from disk."""
        try:
            return _PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error(f"System prompt not found at {_PROMPT_PATH}")
            return "You are EmpathEase, a compassionate AI companion. {EMOTIONAL_STATE}"

    def _build_system_prompt(self, fused_state: FusedEmotionalState) -> str:
        """Inject EMOTIONAL_STATE into system prompt template."""
        emotional_context = self._format_emotional_state(fused_state)
        return self._prompt_template.replace("{EMOTIONAL_STATE}", emotional_context)

    @staticmethod
    def _format_emotional_state(state: FusedEmotionalState) -> str:
        """Format FusedEmotionalState as readable context for the LLM."""
        # Top 3 emotions by probability
        sorted_emotions = sorted(
            state.emotions.items(), key=lambda x: x[1], reverse=True
        )[:3]

        ctx = {
            "dominant_emotion": state.dominant_emotion,
            "confidence": state.confidence,
            "valence": state.valence,
            "arousal": state.arousal,
            "top_emotions": {e: round(p, 3) for e, p in sorted_emotions},
            "modalities": state.modalities_used,
            "incongruence": state.incongruence,
        }
        return json.dumps(ctx, indent=2)

    # ------------------------------------------------------------------
    # Groq API (primary)
    # ------------------------------------------------------------------

    async def _call_groq(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Call Groq API for LLM inference."""
        if not self._settings.groq_api_key:
            logger.warning("Groq API key not configured — skipping")
            return None

        try:
            response = await self._http_client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._settings.groq_model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "top_p": 0.9,
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            logger.info("Groq response: %d chars", len(content))
            return content.strip()

        except Exception as e:
            logger.error("Groq API error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Ollama (fallback)
    # ------------------------------------------------------------------

    async def _call_ollama(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Call local Ollama instance as fallback."""
        try:
            response = await self._http_client.post(
                f"{self._settings.ollama_host}/api/chat",
                json={
                    "model": self._settings.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            logger.info("Ollama response: %d chars", len(content))
            return content.strip() if content else None

        except Exception as e:
            logger.error("Ollama error: %s", e)
            return None


# ============================================================================
# Singleton
# ============================================================================

_engine: Optional[TherapyEngine] = None


def get_therapy_engine() -> TherapyEngine:
    """Get or create the singleton therapy engine."""
    global _engine
    if _engine is None:
        _engine = TherapyEngine()
    return _engine
