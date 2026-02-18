"""
Automatic evaluation of LLM-generated outputs.
Evaluates: linguistic accuracy, lexical diversity, domain coverage, code-switching.
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List

from src.language_detector import detect_language, compute_code_switching_rate


# --- Lexical Diversity Metrics ---


def compute_ttr(text: str) -> float:
    """Type-Token Ratio: unique words / total words."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_hapax_ratio(text: str) -> float:
    """Hapax legomena ratio: words appearing only once / total words."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    hapax = sum(1 for count in freq.values() if count == 1)
    return hapax / len(tokens)


def compute_vocabulary_size(text: str) -> int:
    """Number of unique words."""
    tokens = text.lower().split()
    return len(set(tokens))


def compute_word_count(text: str) -> int:
    """Total number of words."""
    return len(text.split())


# --- Output Validity ---


def is_valid_output(text: str, min_length: int = 20) -> bool:
    """Check if output meets minimum length requirement."""
    return len(text.split()) >= min_length


def compute_output_length(text: str) -> Dict:
    """Compute various length metrics."""
    words = text.split()
    sentences = re.split(r"[.!?।\n]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return {
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": len(sentences),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
    }


# --- Evaluate Single Output ---


def evaluate_single(record: dict, config: dict) -> dict:
    """
    Evaluate a single generated output.

    Args:
        record: Dict with keys: output, language, task_type, etc.
        config: Experiment config

    Returns:
        Dict with all evaluation scores
    """
    text = record.get("output", "")
    lang_code = record.get("language", "")
    min_length = config["evaluation"].get("min_output_length", 20)

    # Colonial language mapping
    colonial_map = {"hau": "en", "fon": "fr"}
    colonial_lang = colonial_map.get(lang_code, "en")

    # Validity
    valid = is_valid_output(text, min_length)

    # Length metrics
    length_metrics = compute_output_length(text)

    # Lexical diversity
    ttr = compute_ttr(text)
    hapax = compute_hapax_ratio(text)
    vocab_size = compute_vocabulary_size(text)

    # Language detection
    lang_detection = detect_language(text)

    # Code-switching
    # Note: langdetect may not reliably detect Hausa/Fongbe,
    # so we primarily check for colonial language presence
    cs_metrics = compute_code_switching_rate(text, lang_code, colonial_lang)

    return {
        "prompt_id": record.get("prompt_id"),
        "task_type": record.get("task_type"),
        "subtask": record.get("subtask"),
        "language": lang_code,
        "model": record.get("model"),
        "is_valid": valid,
        # Length
        "word_count": length_metrics["word_count"],
        "char_count": length_metrics["char_count"],
        "sentence_count": length_metrics["sentence_count"],
        "avg_sentence_length": length_metrics["avg_sentence_length"],
        # Lexical diversity
        "ttr": ttr,
        "hapax_ratio": hapax,
        "vocabulary_size": vocab_size,
        # Language fidelity
        "detected_language": lang_detection["detected"],
        "lang_confidence": lang_detection["confidence"],
        "all_detected_langs": lang_detection["all_langs"],
        # Code-switching
        "code_switching_rate": cs_metrics["code_switching_rate"],
        "colonial_sentences": cs_metrics["colonial_sentences"],
        "target_sentences": cs_metrics["target_sentences"],
        "total_sentences_analyzed": cs_metrics["total_sentences"],
    }


# --- Evaluate All Outputs ---


def evaluate_all_outputs(config: dict):
    """
    Evaluate all raw outputs and save evaluation results.
    Reads from outputs/raw/, writes to outputs/evaluated/.
    """
    raw_dir = config["paths"]["raw_output_dir"]
    eval_dir = config["paths"]["eval_output_dir"]

    all_evals = []

    for model_key in os.listdir(raw_dir):
        model_dir = os.path.join(raw_dir, model_key)
        if not os.path.isdir(model_dir):
            continue

        for lang_folder in os.listdir(model_dir):
            lang_dir = os.path.join(model_dir, lang_folder)
            if not os.path.isdir(lang_dir):
                continue

            lang_code = {"hausa": "hau", "fongbe": "fon"}.get(lang_folder)
            if not lang_code:
                continue

            for task_type in os.listdir(lang_dir):
                task_dir = os.path.join(lang_dir, task_type)
                if not os.path.isdir(task_dir):
                    continue

                # Output directory
                eval_task_dir = os.path.join(
                    eval_dir, model_key, lang_folder, task_type
                )
                os.makedirs(eval_task_dir, exist_ok=True)

                task_evals = []
                for filename in sorted(os.listdir(task_dir)):
                    if not filename.endswith(".json") or filename.startswith("_"):
                        continue

                    filepath = os.path.join(task_dir, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        record = json.load(f)

                    eval_result = evaluate_single(record, config)
                    task_evals.append(eval_result)

                # Save task-level evaluations
                eval_path = os.path.join(eval_task_dir, "evaluations.json")
                with open(eval_path, "w", encoding="utf-8") as f:
                    json.dump(task_evals, f, ensure_ascii=False, indent=2)

                all_evals.extend(task_evals)
                print(
                    f"Evaluated: {model_key}/{lang_folder}/{task_type} "
                    f"({len(task_evals)} outputs)"
                )

    return all_evals
