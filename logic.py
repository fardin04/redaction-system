# logic.py
import spacy
import logging
import sys
import re
from typing import Tuple, List

# Presidio imports
try:
    from presidio_analyzer import (
        AnalyzerEngine,
        PatternRecognizer,
        Pattern,
        RecognizerRegistry,
    )
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_analyzer = None
_anonymizer = AnonymizerEngine()

USE_SMALL_MODEL = False  # toggle for demo speed

def get_nlp_config():
    """Configure spaCy NLP engine"""
    model_name = "en_core_web_lg"
    if USE_SMALL_MODEL:
        model_name = "en_core_web_sm"
    try:
        if not spacy.util.is_package(model_name):
            logger.info(f"Downloading spaCy model: {model_name}...")
            spacy.cli.download(model_name)
        return {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": model_name}]}
    except Exception:
        return {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}

def add_custom_recognizers(registry: RecognizerRegistry):
    """Add regex-based recognizers on top of built-in ones"""
    
    # Credit card
    cc_pattern = Pattern(name="cc_grouped_strict", regex=r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b", score=1.0)
    registry.add_recognizer(PatternRecognizer(supported_entity="CREDIT_CARD", patterns=[cc_pattern]))

    # Email
    email_pattern = Pattern(name="email_simple", regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", score=0.99)
    registry.add_recognizer(PatternRecognizer(supported_entity="EMAIL_ADDRESS", patterns=[email_pattern]))

    # URL
    url_pattern = Pattern(name="url_http", regex=r"\b(?:https?://|http?://|www\.)[^\s<>\"']+\b", score=0.98)
    registry.add_recognizer(PatternRecognizer(supported_entity="URL", patterns=[url_pattern]))

    # === FIXED: LOCATION - Only capture the value AFTER the label ===
    location_patterns = [
        # Capture only the city name after "Location:", "City:", etc.
        Pattern(
            name="location_after_label",
            regex=r"(?<=\b(?:Location|City|Area|Place)\s*:\s*)([A-Z][a-z]+(?:\s[A-Za-z]+)*)",
            score=0.99
        ),
    ]
    registry.add_recognizer(PatternRecognizer(supported_entity="LOCATION", patterns=location_patterns))

    # IPv4
    ip_pattern = Pattern(name="ipv4_strict", regex=r"\b(?:25[0-5]|2[0-4]\d|1?\d{1,2})(?:\.(?:25[0-5]|2[0-4]\d|1?\d{1,2})){3}\b", score=0.99)
    registry.add_recognizer(PatternRecognizer(supported_entity="IP_ADDRESS", patterns=[ip_pattern]))

    # Phone
    phone_pattern = Pattern(name="phone_common", regex=r"(?:\b|\+)\d{1,3}[-.\s]?(?:\(?\d{2,4}\)?[-.\s]?)\d{3,4}[-.\s]?\d{3,4}\b", score=0.95)
    registry.add_recognizer(PatternRecognizer(supported_entity="PHONE_NUMBER", patterns=[phone_pattern]))

    # TIME
    time_pattern = Pattern(
        name="time_patterns",
        regex=r"\b(?:[01]?\d|2[0-3])(?::[0-5]\d){1,2}(?:\s*(?:AM|PM|am|pm|A\.M\.|P\.M\.))?\b|\b(?:[1-9]|1[0-2])(?:\s?:\s?[0-5]\d)?\s*(?:AM|PM|am|pm)\b",
        score=0.92
    )
    registry.add_recognizer(PatternRecognizer(supported_entity="TIME", patterns=[time_pattern]))

    # DATE
    date_pattern = Pattern(
        name="date_patterns",
        regex=r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?)\b(?!\s*[:0-9])",
        score=0.92
    )
    registry.add_recognizer(PatternRecognizer(supported_entity="DATE", patterns=[date_pattern]))

    # ZIP code
    zip_pattern = Pattern(name="zip_us_uk", regex=r"\b\d{5}(?:-\d{4})?\b|\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", score=0.6)
    registry.add_recognizer(PatternRecognizer(supported_entity="ZIP_CODE", patterns=[zip_pattern]))

    # File names
    file_pattern = Pattern(name="file_name_simple", regex=r"\b[\w\-\.]+\.(?:json|jpg|jpeg|png|gif|bmp|svg|xml|xlsx|docx|pdf|txt|log|csv|yaml|yml)\b", score=0.7)
    registry.add_recognizer(PatternRecognizer(supported_entity="FILE_NAME", patterns=[file_pattern]))

    # API key
    api_pattern = Pattern(name="api_key_value", regex=r"\b(?:AKIA|sk_live_|sk_test_)[A-Za-z0-9\-_]{8,}\b|(?<![A-Za-z0-9])(?:[A-Za-z0-9]{32,})\b", score=0.8)
    registry.add_recognizer(PatternRecognizer(supported_entity="API_KEY", patterns=[api_pattern]))

    # MAC address
    mac_pattern = Pattern(name="mac_address_format", regex=r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b", score=0.99)
    registry.add_recognizer(PatternRecognizer(supported_entity="MAC_ADDRESS", patterns=[mac_pattern]))

    # === FIXED: PERSON - Only capture names AFTER labels ===
    person_patterns = [
        # After "Name:", "User:", etc. - only capture the name part
        Pattern(
            name="person_after_label",
            regex=r"(?<=\b(?:Name|User|Student|Subject|Contact)\s*:\s*)([A-Z][a-z]+(?:\s[A-Z]['\u2019a-z]+){0,3})",
            score=0.99
        ),
        # Lowercase names in specific contexts
        Pattern(
            name="lowercase_name_context",
            regex=r"(?:(?<=\bname is )|(?<=\bcalled )|(?<=\bby )|(?<=\breported by ))([a-z][a-z]+(?:\s[a-z][a-z]+){0,2})\b",
            score=0.65
        ),
        # Quoted names
        Pattern(
            name="quoted_name", 
            regex=r"(?<=[\"'])([A-Za-z][A-Za-z]+(?:\s[A-Za-z]+){0,2})(?=[\"'])", 
            score=0.4
        ),
    ]
    registry.add_recognizer(PatternRecognizer(supported_entity="PERSON", patterns=person_patterns))

    # Street address
    street_pattern = Pattern(
        name="smart_address",
        regex=r"\b\d{1,5}[A-Za-z]?\s+[A-Za-z0-9\.]+(?:\s+[A-Za-z0-9\.]+){0,3}\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Dr|Ln|Way|Ct|Pl)\.?\s*(?:N|S|E|W|NW|NE|SW|SE)?\b",
        score=0.85
    )
    registry.add_recognizer(PatternRecognizer(supported_entity="STREET_ADDRESS", patterns=[street_pattern]))

    # SSN
    ssn_pattern = Pattern(name="ssn_relaxed", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=1.0)
    registry.add_recognizer(PatternRecognizer(supported_entity="US_SSN", patterns=[ssn_pattern]))

    # Passport
    passport_pattern = Pattern(name="passport_alpha_num", regex=r"\b[A-Z]{1,2}\d{7,9}\b", score=0.99)
    registry.add_recognizer(PatternRecognizer(supported_entity="PASSPORT_NUMBER", patterns=[passport_pattern]))

    # Generic IDs
    generic_id_pattern = Pattern(name="generic_id_format", regex=r"\b(?:[A-Z0-9]+[-]){2,3}[A-Z0-9]+\b|\b[A-Z]{3,5}[-]\d{3,5}\b", score=0.6)
    registry.add_recognizer(PatternRecognizer(supported_entity="EMPLOYEE_ID", patterns=[generic_id_pattern]))
    registry.add_recognizer(PatternRecognizer(supported_entity="MEDICAL_ID", patterns=[generic_id_pattern]))


def get_analyzer():
    """Initialize and cache the analyzer"""
    global _analyzer
    if _analyzer is None:
        nlp_config = get_nlp_config()
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        add_custom_recognizers(registry)
        _analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["en"])
    return _analyzer


def resolve_overlaps(results):
    """Resolve overlapping entities based on priority"""
    if not results:
        return []
    priority = {
        "PERSON": 200, "LOCATION": 160, "EMAIL_ADDRESS": 220, "IP_ADDRESS": 210,
        "PHONE_NUMBER": 200, "CREDIT_CARD": 230, "DATE": 170, "TIME": 180,
        "URL": 215, "ZIP_CODE": 150, "FILE_NAME": 80, "API_KEY": 70, "US_SSN": 220,
        "PASSPORT_NUMBER": 180, "EMPLOYEE_ID": 90, "MEDICAL_ID": 90,
        "STREET_ADDRESS": 140, "MAC_ADDRESS": 180
    }
    results.sort(key=lambda r: (r.start, -priority.get(r.entity_type, 0), -r.score))
    final = []
    for r in results:
        overlap = False
        for a in list(final):
            if max(r.start, a.start) < min(r.end, a.end):
                overlap = True
                if priority.get(r.entity_type, 0) > priority.get(a.entity_type, 0):
                    final.remove(a)
                    final.append(r)
                break
        if not overlap:
            final.append(r)
    return sorted(final, key=lambda r: r.start)


def redact_text(text: str, mode: str = "redact") -> Tuple[str, List, List, str]:
    """Redact or mask sensitive entities"""
    if not text:
        return "", [], [], ""

    analyzer = get_analyzer()
    results = analyzer.analyze(text=text, language='en')
    results = resolve_overlaps(results)
    results = [r for r in results if r.entity_type not in ["ORG","ORGANIZATION","PERCENTAGE", "ROUTING_NUMBER", "VIN", "DOCUMENT_ID"]]
    
    # No need for adjustment logic anymore - patterns already handle it!

    # --- TEXT GENERATION LOGIC ---
    output_text = ""
    highlight_text = text  # For visual highlights
    
    if mode == "redact":
        # Generate highlight first (with tags)
        for r in sorted(results, key=lambda x: -x.start):
            highlight_text = highlight_text[:r.start] + f"<{r.entity_type}>" + highlight_text[r.end:]
        
        # Then generate clean output
        output_text = text
        for r in sorted(results, key=lambda x: -x.start):
            before = output_text[:r.start]
            after = output_text[r.end:]
            
            needs_space = (before and not before[-1].isspace() and 
                          after and not after[0].isspace() and after[0] not in ',.;:!?')
            
            output_text = before + (" " if needs_space else "") + after
        
        # Cleanup
        output_text = re.sub(r' +', ' ', output_text)
        output_text = re.sub(r' ([,.;:!?])', r'\1', output_text)
        output_text = re.sub(r'\n +', '\n', output_text)
        output_text = output_text.strip()
        
    elif mode == "tag_angular":
        # Angular tags for highlights
        output_text = text
        for r in sorted(results, key=lambda x: -x.start):
            output_text = output_text[:r.start] + f"<{r.entity_type}>" + output_text[r.end:]
        highlight_text = output_text
        
    else:  # mask mode
        operators = {}
        entity_list = ["CREDIT_CARD","US_SSN","API_KEY","FILE_NAME","STREET_ADDRESS","IP_ADDRESS",
                       "PHONE_NUMBER","EMAIL_ADDRESS","ZIP_CODE","PERSON","LOCATION","GPE","DATE","TIME","URL",
                       "PASSPORT_NUMBER","MAC_ADDRESS","MEDICAL_ID","EMPLOYEE_ID"]

        for et in entity_list:
            operators[et] = OperatorConfig("replace", {"new_value": f"[{et}]"})

        anonymized = _anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
        output_text = anonymized.text
        highlight_text = output_text

    highlights = [f"<{r.entity_type}>" if mode=="redact" else f"[{r.entity_type}]" for r in results]

    return output_text, highlights, results, highlight_text