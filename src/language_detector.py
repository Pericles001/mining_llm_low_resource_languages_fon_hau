"""
Language detection for evaluating whether LLM outputs are in the target language.
Uses langdetect as primary, with token-level analysis for code-switching detection.
"""

from langdetect import detect, detect_langs
from typing import Dict, List, Tuple
import re


def detect_language(text: str) -> Dict:
    """
    Detect the primary language of a text.
    Returns dict with detected language, confidence, and all detected languages.
    """
    if not text or len(text.strip()) < 10:
        return {
            "detected": "unknown",
            "confidence": 0.0,
            "all_langs": [],
        }
    try:
        primary = detect(text)
        all_langs = detect_langs(text)
        lang_scores = {str(lang.lang): lang.prob for lang in all_langs}
        return {
            "detected": primary,
            "confidence": lang_scores.get(primary, 0.0),
            "all_langs": lang_scores,
        }
    except Exception:
        return {
            "detected": "unknown",
            "confidence": 0.0,
            "all_langs": [],
        }


def compute_code_switching_rate(
    text: str, target_lang: str, colonial_lang: str
) -> Dict:
    """
    Estimate code-switching rate by detecting language per sentence.

    Args:
        text: The generated text
        target_lang: Expected language code (e.g., "ha" for Hausa, "fr" for Fongbe detection)
        colonial_lang: Colonial language code (e.g., "en" for English, "fr" for French)

    Returns:
        Dict with code_switching_rate, target_sentences, colonial_sentences, other_sentences
    """
    # Split into sentences
    sentences = re.split(r"[.!?।\n]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences:
        return {
            "code_switching_rate": 0.0,
            "total_sentences": 0,
            "target_sentences": 0,
            "colonial_sentences": 0,
            "other_sentences": 0,
        }

    target_count = 0
    colonial_count = 0
    other_count = 0

    for sent in sentences:
        try:
            detected = detect(sent)
            # langdetect uses ISO 639-1 codes
            if detected in [colonial_lang, colonial_lang[:2]]:
                colonial_count += 1
            elif detected in [target_lang, target_lang[:2]]:
                target_count += 1
            else:
                other_count += 1
        except Exception:
            other_count += 1

    total = len(sentences)
    cs_rate = (colonial_count + other_count) / total if total > 0 else 0.0

    return {
        "code_switching_rate": cs_rate,
        "total_sentences": total,
        "target_sentences": target_count,
        "colonial_sentences": colonial_count,
        "other_sentences": other_count,
    }
