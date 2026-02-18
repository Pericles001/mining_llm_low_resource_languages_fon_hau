"""
Core generator: iterates over all prompts, sends to LLMs, saves raw outputs.
"""

import json
import os
import time
from datetime import datetime
from tqdm import tqdm

from src.models import get_model
from src.prompt_loader import get_all_prompts, get_all_task_types


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def generate_for_condition(
    model_key: str,
    lang_code: str,
    task_type: str,
    config: dict,
    prompts_dir: str,
    output_dir: str,
):
    """
    Generate outputs for one model + language + task type combination.
    Saves each output as a JSON file.
    """
    # Map model keys to folder names
    model_folder = model_key  # e.g., "gemini" or "gpt4o_mini"
    lang_folder = {"hau": "hausa", "fon": "fongbe"}[lang_code]

    # Output directory for this condition
    condition_dir = os.path.join(output_dir, model_folder, lang_folder, task_type)
    ensure_dir(condition_dir)

    # Check what's already generated (for resumability)
    existing = set()
    for f in os.listdir(condition_dir):
        if f.endswith(".json"):
            existing.add(f.replace(".json", ""))

    # Load prompts
    prompts = get_all_prompts(prompts_dir, task_type, lang_code)

    # Initialize model
    model = get_model(model_key, config)
    delay = config["generation"].get("delay_between_calls", 1.0)
    max_retries = config["generation"].get("retry_on_failure", 3)

    results = []
    for prompt_data in tqdm(
        prompts, desc=f"{model_key}/{lang_folder}/{task_type}"
    ):
        prompt_id = prompt_data["id"]

        # Skip if already generated
        if prompt_id in existing:
            print(f"  Skipping {prompt_id} (already exists)")
            continue

        # Generate with retries
        result = None
        for attempt in range(max_retries):
            result = model.generate(prompt_data["prompt"])
            if result["success"]:
                break
            print(
                f"  Retry {attempt + 1}/{max_retries} for {prompt_id}: {result['error']}"
            )
            time.sleep(delay * 2)  # longer delay on retry

        # Build output record
        record = {
            "prompt_id": prompt_id,
            "task_type": task_type,
            "subtask": prompt_data["subtask"],
            "language": lang_code,
            "model": model_key,
            "prompt": prompt_data["prompt"],
            "output": result["text"] if result else "",
            "success": result["success"] if result else False,
            "error": result.get("error"),
            "token_count": result.get("token_count", 0),
            "timestamp": datetime.now().isoformat(),
        }

        # Save individual file
        output_path = os.path.join(condition_dir, f"{prompt_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        results.append(record)
        time.sleep(delay)

    # Save summary for this condition
    summary_path = os.path.join(condition_dir, "_summary.json")
    summary = {
        "model": model_key,
        "language": lang_code,
        "task_type": task_type,
        "total_prompts": len(prompts),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "total_tokens": sum(r["token_count"] for r in results),
        "timestamp": datetime.now().isoformat(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return results


def generate_all(config: dict):
    """Run generation for all models x languages x task types."""
    prompts_dir = config["paths"]["prompts_dir"]
    output_dir = config["paths"]["raw_output_dir"]
    models = list(config["models"].keys())
    languages = [lang["code"] for lang in config["languages"]]
    task_types = get_all_task_types()

    print(f"Models: {models}")
    print(f"Languages: {languages}")
    print(f"Task types: {task_types}")
    print(f"Total conditions: {len(models) * len(languages) * len(task_types)}")
    print(f"Total prompts: {len(models) * len(languages) * len(task_types) * 25}")
    print()

    all_results = []
    for model_key in models:
        for lang_code in languages:
            for task_type in task_types:
                print(f"\n{'='*60}")
                print(f"Generating: {model_key} / {lang_code} / {task_type}")
                print(f"{'='*60}")
                results = generate_for_condition(
                    model_key, lang_code, task_type, config, prompts_dir, output_dir
                )
                all_results.extend(results)

    print(f"\nDone! Total outputs: {len(all_results)}")
    return all_results
