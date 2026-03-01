"""
Crisis Detection Service — deterministic safety middleware.

Pure keyword/pattern matching. Zero LLM dependency.
Fires BEFORE the therapy engine is ever called.

Levels:
    none     — no risk detected
    watch    — mentions of struggle, loneliness; LLM handles normally
    moderate — hopelessness/ideation without plan; gentle check-in
    urgent   — explicit suicidal intent, self-harm, intent to harm others;
               bypass LLM entirely, return hard-coded crisis template
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Crisis assessment result
# ============================================================================

@dataclass
class CrisisAssessment:
    level: str           # none | watch | moderate | urgent
    triggered_by: str    # the keyword/pattern that fired (for audit)
    template_response: str  # hard-coded response (empty for none/watch)


# ============================================================================
# Keyword lexicons (English + Hinglish)
# ============================================================================

# URGENT — immediate risk: suicidal intent, active self-harm, harm to others, overdose
_URGENT_PATTERNS = [
    # English
    r"\b(want|going|plan(?:ning)?|going)\b.{0,20}\b(to )?(kill|end|finish)\b.{0,15}\b(my\s*self|myself|me|life)\b",
    r"\b(suicid|kill\s*my\s*self|end\s*my\s*life|take\s*my\s*life)\b",
    r"\b(cutting|cut\s*my\s*self|hurt\s*my\s*self|self[\s-]*harm)\b",
    r"\b(overdos|took\s*(too\s*many|all)\s*(pills|tablets|medicine))\b",
    r"\b(want|going)\b.{0,15}\b(to )?(hurt|kill|harm)\b.{0,10}\b(someone|people|them|him|her)\b",
    # Hinglish / Hindi (romanized)
    r"\b(mar\s*jaana|mar\s*jaunga|mar\s*jaaungi|marna\s*chahta|marna\s*chahti)\b",
    r"\b(zindagi\s*khatam|jeena\s*nahi|ji\s*nahi\s*sakta|ji\s*nahi\s*sakti)\b",
    r"\b(khud\s*ko\s*maar|apne\s*aap\s*ko\s*khatam|suicide\s*kar)\b",
    r"\b(kisi\s*ko\s*maar|sabko\s*maar)\b",
]

# MODERATE — ideation without plan, persistent hopelessness
_MODERATE_PATTERNS = [
    r"\b(don.*t\s*want\s*to\s*live|no\s*reason\s*to\s*live|better\s*off\s*dead)\b",
    r"\b(hopeless|give\s*up|can.*t\s*go\s*on|nothing\s*matters)\b",
    r"\b(koi\s*faayda\s*nahi|kuch\s*nahi\s*bacha|sab\s*khatam|haar\s*gaya|haar\s*gayi)\b",
    r"\b(jeene\s*ka\s*mann\s*nahi|zindagi\s*se\s*thak|zindagi\s*bekar)\b",
    # Indirect farewell language (high-risk in Indian youth context)
    r"\b(sab\s*ka\s*khayal\s*rakhna)\b",
    r"\b(tumse\s*milke\s*achha\s*laga)\b",
    r"\b(bas\s*itna\s*hi\s*kehna\s*tha)\b",
    r"\b(apna\s*khayal\s*rakhna.*alvida|goodbye.*take\s*care\s*of\s*everyone)\b",
]

# WATCH — struggle signals; LLM handles, but with injected awareness
_WATCH_PATTERNS = [
    r"\b(so\s*lonely|nobody\s*cares|all\s*alone|no\s*one\s*understands)\b",
    r"\b(feel(ing)?\s*trapped|stuck|suffocating|drowning)\b",
    r"\b(can.*t\s*take\s*it|too\s*much\s*pain|breaking\s*down)\b",
    r"\b(akela|koi\s*nahi\s*hai|samajhta\s*nahi\s*koi|bahut\s*dard)\b",
    r"\b(exam\s*fail|result\s*aaya|marks\s*kam|disappoint)\b.{0,20}\b(can.*t|nahi|hopeless|kya\s*karu)\b",
]

_COMPILED = {
    "urgent": [re.compile(p, re.IGNORECASE) for p in _URGENT_PATTERNS],
    "moderate": [re.compile(p, re.IGNORECASE) for p in _MODERATE_PATTERNS],
    "watch": [re.compile(p, re.IGNORECASE) for p in _WATCH_PATTERNS],
}


# ============================================================================
# Hard-coded crisis templates
# ============================================================================

_CRISIS_TEMPLATES = {
    "urgent": (
        "Main samajh sakta/sakti hoon ki aap bahut mushkil waqt se guzar rahe hain. "
        "Aap akele nahi hain.\n\n"
        "Please abhi kisi se baat karein:\n"
        "- Vandrevala Foundation: 1860-2662-345 (24/7)\n"
        "- iCall: 9152987821\n"
        "- AASRA: 9820466726\n\n"
        "Kya aap abhi safe hain? Agar aapko turant madad chahiye, "
        "toh please 112 dial karein."
    ),
    "moderate": (
        "Jo aap feel kar rahe hain, woh valid hai — aur aapko akele face "
        "nahi karna hai. Kya aap mujhe thoda aur batana chahenge?\n\n"
        "Agar aap chahein toh kisi trained counsellor se baat kar sakte hain — "
        "Vandrevala Foundation (1860-2662-345) par 24/7 call kar sakte hain. "
        "Woh samjhenge."
    ),
}


# ============================================================================
# Public API
# ============================================================================

def assess_crisis(
    text: str,
    fused_state=None,
) -> CrisisAssessment:
    """
    Assess crisis level from user text.

    Pure function — no LLM, no side effects.
    Checks urgent → moderate → watch in priority order.

    Args:
        text: User's message text
        fused_state: Optional FusedEmotionalState (reserved for future use)

    Returns:
        CrisisAssessment with level, trigger, and optional template
    """
    if not text or not text.strip():
        return CrisisAssessment(level="none", triggered_by="", template_response="")

    text_clean = text.strip()

    # Check in priority order: urgent → moderate → watch
    for level in ("urgent", "moderate", "watch"):
        for pattern in _COMPILED[level]:
            match = pattern.search(text_clean)
            if match:
                triggered = match.group(0)
                template = _CRISIS_TEMPLATES.get(level, "")

                logger.warning(
                    "Crisis detected: level=%s, trigger='%s'",
                    level, triggered,
                )

                return CrisisAssessment(
                    level=level,
                    triggered_by=triggered,
                    template_response=template,
                )

    return CrisisAssessment(level="none", triggered_by="", template_response="")
