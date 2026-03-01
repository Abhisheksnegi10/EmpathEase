"""
Standalone test runner for crisis service + therapy engine.
Run:  $env:PYTHONIOENCODING='utf-8'; python tests/run_therapy_tests.py
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.crisis import assess_crisis, CrisisAssessment
from app.schemas.emotion import FusedEmotionalState
from app.services.therapy import TherapyEngine

passed = 0
failed = 0
errors = []


def check(name, condition, msg=""):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        errors.append(f"  FAIL {name}: {msg}")
        print(f"  FAIL {name}: {msg}")


# ============================================================================
# Crisis Detection Tests
# ============================================================================

print("\n--- Crisis: Urgent Triggers (English) ---")

r = assess_crisis("I want to kill myself")
check("kill myself = urgent", r.level == "urgent")

r = assess_crisis("I'm planning to end my life tonight")
check("end my life = urgent", r.level == "urgent")

r = assess_crisis("I've been cutting myself")
check("cutting myself = urgent", r.level == "urgent")

r = assess_crisis("I took too many pills")
check("too many pills = urgent", r.level == "urgent")

r = assess_crisis("I want to hurt someone")
check("hurt someone = urgent", r.level == "urgent")

print("\n--- Crisis: Urgent Triggers (Hinglish) ---")

r = assess_crisis("main mar jaana chahta hoon")
check("mar jaana chahta = urgent", r.level == "urgent")

r = assess_crisis("zindagi khatam karna hai")
check("zindagi khatam = urgent", r.level == "urgent")

r = assess_crisis("khud ko maar dunga")
check("khud ko maar = urgent", r.level == "urgent")

print("\n--- Crisis: Moderate Triggers ---")

r = assess_crisis("I don't want to live anymore")
check("don't want to live = moderate", r.level == "moderate")

r = assess_crisis("nothing matters, I give up")
check("nothing matters give up = moderate", r.level == "moderate")

r = assess_crisis("koi faayda nahi hai kuch karne ka")
check("koi faayda nahi = moderate", r.level == "moderate")

print("\n--- Crisis: Indirect Farewell (Indian Youth) ---")

r = assess_crisis("sab ka khayal rakhna")
check("sab ka khayal rakhna = moderate", r.level == "moderate")

r = assess_crisis("tumse milke achha laga")
check("tumse milke achha laga = moderate", r.level == "moderate")

r = assess_crisis("bas itna hi kehna tha")
check("bas itna hi kehna tha = moderate", r.level == "moderate")

print("\n--- Crisis: Watch Triggers ---")

r = assess_crisis("I feel so lonely, nobody cares")
check("so lonely nobody cares = watch", r.level == "watch")

r = assess_crisis("I'm feeling trapped and suffocating")
check("feeling trapped = watch", r.level == "watch")

r = assess_crisis("akela feel hota hai, koi nahi hai")
check("akela koi nahi hai = watch", r.level == "watch")

print("\n--- Crisis: Clean Text (No Trigger) ---")

r = assess_crisis("I had a good day today")
check("positive text = none", r.level == "none")

r = assess_crisis("Can you help me with meditation?")
check("neutral request = none", r.level == "none")

r = assess_crisis("Aaj mausam achha hai")
check("neutral Hindi = none", r.level == "none")

r = assess_crisis("")
check("empty text = none", r.level == "none")

r = assess_crisis("   ")
check("whitespace = none", r.level == "none")

print("\n--- Crisis: Templates ---")

r = assess_crisis("I want to kill myself")
check("urgent has template", len(r.template_response) > 50)
check("urgent template has helpline", "1860-2662-345" in r.template_response)

r = assess_crisis("nothing matters")
check("moderate has template", len(r.template_response) > 20)

r = assess_crisis("I feel lonely")
check("watch has no template override", r.template_response == "")

r = assess_crisis("hello")
check("none has empty template", r.template_response == "")

print("\n--- Crisis: Priority Order ---")

# If text matches both urgent and moderate, urgent wins
r = assess_crisis("I want to kill myself because nothing matters")
check("urgent wins over moderate", r.level == "urgent")


# ============================================================================
# Therapy Engine Tests
# ============================================================================

print("\n--- Therapy: Prompt Building ---")

engine = TherapyEngine()

check("prompt template loaded", len(engine._prompt_template) > 100)
check("has EMOTIONAL_STATE placeholder", "{EMOTIONAL_STATE}" in engine._prompt_template)

# Build system prompt
state = FusedEmotionalState(
    dominant_emotion="sadness",
    confidence=0.85,
    valence=-0.6,
    arousal=0.3,
    modalities_used=["text", "voice"],
)
prompt = engine._build_system_prompt(state)
check("EMOTIONAL_STATE injected", "sadness" in prompt)
check("placeholder replaced", "{EMOTIONAL_STATE}" not in prompt)

# Format emotional state
ctx = engine._format_emotional_state(state)
check("emotional context is JSON", '"dominant_emotion"' in ctx)
check("top_emotions present", '"top_emotions"' in ctx)

print("\n--- Therapy: LLM Config ---")

from app.config import get_settings
s = get_settings()
check("groq model = llama-3.3", "3.3" in s.groq_model)
check("ollama host configured", "localhost" in s.ollama_host)
check("sarvam stt = saaras:v3", s.sarvam_stt_model == "saaras:v3")
check("sarvam tts = bulbul:v2", s.sarvam_tts_model == "bulbul:v2")
check("sarvam speaker = anushka", s.sarvam_tts_speaker == "anushka")
check("crisis_threshold = 0.7", s.crisis_threshold == 0.7)
check("working_memory_ttl = 1800", s.working_memory_ttl == 1800)
check("max_conversation_turns = 20", s.max_conversation_turns == 20)

print("\n--- Therapy: Schema ---")

state = FusedEmotionalState()
check("crisis_level default = none", state.crisis_level == "none")
state.crisis_level = "urgent"
check("crisis_level mutable", state.crisis_level == "urgent")
d = state.model_dump()
check("crisis_level serializes", d["crisis_level"] == "urgent")


# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print("\nFailures:")
    for e in errors:
        print(e)
print(f"{'='*60}")
sys.exit(0 if failed == 0 else 1)
