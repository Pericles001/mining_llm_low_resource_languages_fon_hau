"""
Loads prompt templates from JSON files and formats them for each target language.
"""

import json
import os
from typing import List, Dict


# Language-specific substitutions
LANGUAGE_CONFIG = {
    "hau": {
        "language": "Hausa",
        "language_culture": "Hausa",
        "colonial_language": "English",
    },
    "fon": {
        "language": "Fongbe",
        "language_culture": "Fongbe (Benin)",
        "colonial_language": "French",
    },
}


def load_prompts(prompts_dir: str, task_type: str) -> dict:
    """Load prompts from a JSON file for a given task type."""
    filepath = os.path.join(prompts_dir, f"{task_type}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_prompt(template: str, lang_code: str, prompt_data: dict = None) -> str:
    """
    Format a prompt template with language-specific values.

    Handles {language}, {language_culture}, {colonial_language},
    and word list substitutions for constrained generation.
    """
    lang_config = LANGUAGE_CONFIG[lang_code]
    formatted = template.format(
        language=lang_config["language"],
        language_culture=lang_config["language_culture"],
        colonial_language=lang_config["colonial_language"],
    )

    # Handle constrained generation word lists
    if prompt_data and "word_lists" in prompt_data:
        lang_name = lang_config["language"]
        if lang_name in prompt_data["word_lists"]:
            word_list = prompt_data["word_lists"][lang_name]
            # Replace any {word_list_N} placeholders
            for i in range(1, 10):
                placeholder = f"{{word_list_{i}}}"
                if placeholder in formatted:
                    formatted = formatted.replace(placeholder, word_list)
    return formatted


def get_all_prompts(
    prompts_dir: str, task_type: str, lang_code: str
) -> List[Dict[str, str]]:
    """
    Get all formatted prompts for a task type and language.

    Returns list of dicts: {id, prompt, subtask, task_type}
    """
    data = load_prompts(prompts_dir, task_type)
    prompts = []
    for p in data["prompts"]:
        formatted_prompt = format_prompt(p["template"], lang_code, p)
        prompts.append(
            {
                "id": p["id"],
                "prompt": formatted_prompt,
                "subtask": p["subtask"],
                "task_type": task_type,
            }
        )
    return prompts


def get_all_task_types() -> List[str]:
    """Return list of all 6 task types."""
    return [
        "creative_writing",
        "functional_text",
        "structured_knowledge",
        "dialogue",
        "topic_switching",
        "constrained_generation",
    ]
