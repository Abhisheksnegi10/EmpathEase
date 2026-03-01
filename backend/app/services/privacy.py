"""
PII Scrubbing Service - Privacy & Data Protection (FR-013).

Implements PII detection and redaction using Microsoft Presidio.

Redacts:
- Names (PERSON)
- Phone numbers (PHONE_NUMBER)
- Email addresses (EMAIL_ADDRESS)
- Locations (LOCATION)
- SSN, credit cards (US_SSN, CREDIT_CARD)
- IP addresses (IP_ADDRESS)
- Dates of birth (DATE_TIME)

Applied BEFORE storing in long-term memory for GDPR compliance.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of PII that can be detected."""
    PERSON = "PERSON"
    PHONE_NUMBER = "PHONE_NUMBER"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    LOCATION = "LOCATION"
    US_SSN = "US_SSN"
    CREDIT_CARD = "CREDIT_CARD"
    IP_ADDRESS = "IP_ADDRESS"
    DATE_TIME = "DATE_TIME"
    URL = "URL"
    IBAN_CODE = "IBAN_CODE"
    NRP = "NRP"  # Nationality, Religion, Political group


# Placeholder tokens for redaction
REDACTION_MAP = {
    PIIType.PERSON: "[PERSON]",
    PIIType.PHONE_NUMBER: "[PHONE]",
    PIIType.EMAIL_ADDRESS: "[EMAIL]",
    PIIType.LOCATION: "[LOCATION]",
    PIIType.US_SSN: "[SSN]",
    PIIType.CREDIT_CARD: "[CREDIT_CARD]",
    PIIType.IP_ADDRESS: "[IP]",
    PIIType.DATE_TIME: "[DATE]",
    PIIType.URL: "[URL]",
    PIIType.IBAN_CODE: "[IBAN]",
    PIIType.NRP: "[NRP]",
}


@dataclass
class PIIEntity:
    """Detected PII entity."""
    entity_type: str
    start: int
    end: int
    text: str
    score: float


@dataclass
class ScrubResult:
    """Result of PII scrubbing."""
    original_text: str
    scrubbed_text: str
    entities_found: List[PIIEntity]
    pii_count: int


class PIIScrubber:
    """
    PII detection and scrubbing using Microsoft Presidio.
    
    Falls back to regex-based detection if Presidio is not installed.
    """
    
    def __init__(
        self,
        entities_to_detect: Optional[List[str]] = None,
        language: str = "en",
    ):
        """
        Initialize PII scrubber.
        
        Args:
            entities_to_detect: List of PII types to detect (None = all)
            language: Language for NER detection
        """
        self.language = language
        self.entities = entities_to_detect or [
            PIIType.PERSON,
            PIIType.PHONE_NUMBER,
            PIIType.EMAIL_ADDRESS,
            PIIType.LOCATION,
            PIIType.US_SSN,
            PIIType.CREDIT_CARD,
            PIIType.IP_ADDRESS,
        ]
        
        self._analyzer = None
        self._anonymizer = None
        self._use_presidio = False
        
        # Try to load Presidio
        self._initialize_presidio()
        
        logger.info(f"PIIScrubber initialized (presidio: {self._use_presidio})")
    
    def _initialize_presidio(self) -> None:
        """Initialize Presidio analyzer and anonymizer."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._use_presidio = True
            
            logger.info("Presidio initialized successfully")
            
        except ImportError:
            logger.warning(
                "Presidio not installed. Using regex fallback. "
                "Install with: pip install presidio-analyzer presidio-anonymizer"
            )
            self._use_presidio = False
    
    def scrub(
        self,
        text: str,
        return_entities: bool = True,
    ) -> ScrubResult:
        """
        Scrub PII from text.
        
        Args:
            text: Input text to scrub
            return_entities: Include detected entities in result
        
        Returns:
            ScrubResult with scrubbed text and entity list.
        """
        if not text or not text.strip():
            return ScrubResult(
                original_text=text,
                scrubbed_text=text,
                entities_found=[],
                pii_count=0,
            )
        
        if self._use_presidio:
            return self._scrub_with_presidio(text, return_entities)
        else:
            return self._scrub_with_regex(text, return_entities)
    
    def _scrub_with_presidio(
        self,
        text: str,
        return_entities: bool,
    ) -> ScrubResult:
        """Scrub using Presidio."""
        from presidio_anonymizer.entities import OperatorConfig
        
        # Analyze
        results = self._analyzer.analyze(
            text=text,
            entities=[e.value if isinstance(e, PIIType) else e for e in self.entities],
            language=self.language,
        )
        
        # Convert to our entity format
        entities = []
        if return_entities:
            for r in results:
                entities.append(PIIEntity(
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    text=text[r.start:r.end],
                    score=r.score,
                ))
        
        # Anonymize with custom operators
        operators = {}
        for entity_type in self.entities:
            key = entity_type.value if isinstance(entity_type, PIIType) else entity_type
            placeholder = REDACTION_MAP.get(
                PIIType(key) if key in [e.value for e in PIIType] else None,
                f"[{key}]"
            )
            operators[key] = OperatorConfig("replace", {"new_value": placeholder})
        
        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )
        
        return ScrubResult(
            original_text=text,
            scrubbed_text=anonymized.text,
            entities_found=entities,
            pii_count=len(results),
        )
    
    def _scrub_with_regex(
        self,
        text: str,
        return_entities: bool,
    ) -> ScrubResult:
        """Fallback regex-based scrubbing."""
        scrubbed = text
        entities = []
        
        # Define regex patterns
        patterns = [
            # Email
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', PIIType.EMAIL_ADDRESS),
            # Phone (various formats)
            (r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b', PIIType.PHONE_NUMBER),
            # SSN
            (r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', PIIType.US_SSN),
            # Credit card
            (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', PIIType.CREDIT_CARD),
            # IP address
            (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', PIIType.IP_ADDRESS),
            # URL
            (r'https?://[^\s]+', PIIType.URL),
        ]
        
        for pattern, pii_type in patterns:
            matches = list(re.finditer(pattern, scrubbed))
            for match in reversed(matches):  # Reverse to maintain positions
                placeholder = REDACTION_MAP.get(pii_type, f"[{pii_type.value}]")
                
                if return_entities:
                    entities.append(PIIEntity(
                        entity_type=pii_type.value,
                        start=match.start(),
                        end=match.end(),
                        text=match.group(),
                        score=1.0,
                    ))
                
                scrubbed = scrubbed[:match.start()] + placeholder + scrubbed[match.end():]
        
        # Simple name detection (capitalized words after certain patterns)
        name_patterns = [
            r"(?:my name is|I'm|I am|call me)\s+([A-Z][a-z]+)",
            r"(?:my|our)\s+(?:sister|brother|mom|dad|mother|father|friend|boss)\s+([A-Z][a-z]+)",
        ]
        
        for pattern in name_patterns:
            matches = list(re.finditer(pattern, scrubbed, re.IGNORECASE))
            for match in reversed(matches):
                if match.group(1):
                    name = match.group(1)
                    name_start = match.start(1)
                    name_end = match.end(1)
                    
                    if return_entities:
                        entities.append(PIIEntity(
                            entity_type=PIIType.PERSON.value,
                            start=name_start,
                            end=name_end,
                            text=name,
                            score=0.8,
                        ))
                    
                    scrubbed = scrubbed[:name_start] + "[PERSON]" + scrubbed[name_end:]
        
        # Reverse entities to maintain original order
        entities.reverse()
        
        return ScrubResult(
            original_text=text,
            scrubbed_text=scrubbed,
            entities_found=entities,
            pii_count=len(entities),
        )
    
    def detect(self, text: str) -> List[PIIEntity]:
        """
        Detect PII without redacting.
        
        Args:
            text: Input text
        
        Returns:
            List of detected PII entities.
        """
        result = self.scrub(text, return_entities=True)
        return result.entities_found
    
    def has_pii(self, text: str) -> bool:
        """
        Quick check if text contains any PII.
        
        Args:
            text: Input text
        
        Returns:
            True if PII detected.
        """
        result = self.scrub(text, return_entities=False)
        return result.pii_count > 0
    
    def scrub_dict(
        self,
        data: Dict,
        keys_to_scrub: Optional[List[str]] = None,
    ) -> Dict:
        """
        Scrub PII from dictionary values.
        
        Args:
            data: Dictionary with string values
            keys_to_scrub: Specific keys to scrub (None = all string values)
        
        Returns:
            Dictionary with scrubbed values.
        """
        result = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                if keys_to_scrub is None or key in keys_to_scrub:
                    result[key] = self.scrub(value, return_entities=False).scrubbed_text
                else:
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = self.scrub_dict(value, keys_to_scrub)
            elif isinstance(value, list):
                result[key] = [
                    self.scrub(item, return_entities=False).scrubbed_text
                    if isinstance(item, str) else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result


# Singleton
_scrubber: Optional[PIIScrubber] = None


def get_scrubber() -> PIIScrubber:
    """Get singleton PII scrubber instance."""
    global _scrubber
    if _scrubber is None:
        _scrubber = PIIScrubber()
    return _scrubber


def scrub_pii(text: str) -> str:
    """
    Convenience function to scrub PII from text.
    
    Args:
        text: Input text
    
    Returns:
        Text with PII redacted.
    """
    return get_scrubber().scrub(text, return_entities=False).scrubbed_text


def detect_pii(text: str) -> List[PIIEntity]:
    """
    Convenience function to detect PII in text.
    
    Args:
        text: Input text
    
    Returns:
        List of PII entities.
    """
    return get_scrubber().detect(text)
