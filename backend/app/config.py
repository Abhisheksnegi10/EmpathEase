"""
Application configuration using Pydantic settings.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "EmpathEase"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-me-in-production"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000



    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "episodic_memory"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "dev_password_change_in_prod"

    # JWT Authentication
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # ML Models
    facial_model_path: str = "app/ml/models/facial_affect.onnx"
    text_emotion_model: str = "distilbert-base-uncased-finetuned-goemotions"

    # Fusion — STT-aware cascade
    fusion_text_weight_high: float = 0.60    # STT ≥ 0.85
    fusion_text_weight_medium: float = 0.40  # STT 0.65–0.85
    fusion_text_weight_low: float = 0.00     # STT < 0.65
    fusion_face_weight: float = 0.30         # stable across tiers (0.10 at high tier filled by voice delta)
    fusion_voice_weight_high: float = 0.10
    fusion_voice_weight_medium: float = 0.30
    fusion_voice_weight_low: float = 0.60
    fusion_incongruence_threshold: float = 0.3  # valence-zone based

    # LLM — Groq primary, Ollama fallback
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b"

    # Sarvam AI — STT/TTS
    sarvam_api_key: str = ""
    sarvam_stt_model: str = "saaras:v3"
    sarvam_tts_model: str = "bulbul:v2"
    sarvam_tts_speaker: str = "meera"
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_tts_url: str = "https://api.sarvam.ai/text-to-speech"

    # Session
    working_memory_ttl: int = 1800        # 30 min
    max_conversation_turns: int = 20

    # Safety
    crisis_threshold: float = 0.7
    enable_pii_scrubbing: bool = True

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
