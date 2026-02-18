#!/usr/bin/env python3
"""
CLI entry point for evaluating generated outputs.

Usage:
    python scripts/run_evaluation.py --all
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config
from src.evaluator import evaluate_all_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-generated outputs"
    )
    parser.add_argument(
        "--all", action="store_true", help="Evaluate all generated outputs"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.all:
        print("Evaluating all outputs...")
        results = evaluate_all_outputs(config)
        print(f"\nTotal evaluations: {len(results)}")
        # Quick summary
        valid = sum(1 for r in results if r["is_valid"])
        print(f"Valid outputs: {valid}/{len(results)} ({100*valid/max(len(results),1):.1f}%)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
