#!/usr/bin/env python3
"""
CLI entry point for generating LLM outputs.

Usage:
    python scripts/run_generation.py --all
    python scripts/run_generation.py --model gemini --lang hau --task creative_writing
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, ensure_all_dirs
from src.generator import generate_for_condition, generate_all
from src.prompt_loader import get_all_task_types


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM outputs for mining experiment"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all models x languages x tasks"
    )
    parser.add_argument(
        "--model",
        choices=["gemini", "gpt4o_mini"],
        help="Specific model to use",
    )
    parser.add_argument(
        "--lang", choices=["hau", "fon"], help="Specific language"
    )
    parser.add_argument(
        "--task",
        choices=get_all_task_types(),
        help="Specific task type",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    ensure_all_dirs(config)

    if args.all:
        print("Running ALL conditions...")
        print(f"  Models: {list(config['models'].keys())}")
        print(f"  Languages: {[l['code'] for l in config['languages']]}")
        print(f"  Tasks: {get_all_task_types()}")
        print(f"  Total prompts: {2 * 2 * 6 * 25} = 600")
        print()
        generate_all(config)

    elif args.model and args.lang and args.task:
        print(f"Running: {args.model} / {args.lang} / {args.task}")
        generate_for_condition(
            model_key=args.model,
            lang_code=args.lang,
            task_type=args.task,
            config=config,
            prompts_dir=config["paths"]["prompts_dir"],
            output_dir=config["paths"]["raw_output_dir"],
        )

    else:
        print("Use --all or specify --model, --lang, and --task")
        parser.print_help()


if __name__ == "__main__":
    main()
