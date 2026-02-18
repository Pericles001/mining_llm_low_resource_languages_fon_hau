#!/usr/bin/env python3
"""
Aggregate evaluation results and generate tables/figures for the paper.

Usage:
    python scripts/run_analysis.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.utils import load_config


def load_all_evaluations(eval_dir: str) -> pd.DataFrame:
    """Load all evaluation JSONs into a single DataFrame."""
    records = []
    for model in os.listdir(eval_dir):
        model_dir = os.path.join(eval_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for lang in os.listdir(model_dir):
            lang_dir = os.path.join(model_dir, lang)
            if not os.path.isdir(lang_dir):
                continue
            for task in os.listdir(lang_dir):
                eval_file = os.path.join(lang_dir, task, "evaluations.json")
                if os.path.exists(eval_file):
                    with open(eval_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        records.extend(data)

    return pd.DataFrame(records)


def generate_summary_table(df: pd.DataFrame, results_dir: str):
    """Generate main summary table: metrics by model x language x task."""
    summary = (
        df.groupby(["model", "language", "task_type"])
        .agg(
            {
                "is_valid": "mean",
                "word_count": "mean",
                "ttr": "mean",
                "hapax_ratio": "mean",
                "vocabulary_size": "mean",
                "code_switching_rate": "mean",
                "lang_confidence": "mean",
            }
        )
        .round(3)
    )

    # Save CSV
    csv_path = os.path.join(results_dir, "summary.csv")
    summary.to_csv(csv_path)
    print(f"Saved summary table: {csv_path}")

    # Generate LaTeX table
    latex_path = os.path.join(results_dir, "tables", "summary_table.tex")
    summary.to_latex(latex_path, float_format="%.3f")
    print(f"Saved LaTeX table: {latex_path}")

    return summary


def generate_language_comparison(df: pd.DataFrame, results_dir: str):
    """Compare metrics across languages, aggregated by task type."""
    comparison = (
        df.groupby(["language", "task_type"])
        .agg(
            {
                "word_count": ["mean", "std"],
                "ttr": ["mean", "std"],
                "code_switching_rate": ["mean", "std"],
                "vocabulary_size": ["mean", "std"],
            }
        )
        .round(3)
    )

    csv_path = os.path.join(results_dir, "language_comparison.csv")
    comparison.to_csv(csv_path)
    print(f"Saved language comparison: {csv_path}")
    return comparison


def generate_model_comparison(df: pd.DataFrame, results_dir: str):
    """Compare metrics across models, aggregated by language."""
    comparison = (
        df.groupby(["model", "language"])
        .agg(
            {
                "is_valid": "mean",
                "word_count": "mean",
                "ttr": "mean",
                "code_switching_rate": "mean",
                "vocabulary_size": "sum",  # total unique words extracted
            }
        )
        .round(3)
    )

    csv_path = os.path.join(results_dir, "model_comparison.csv")
    comparison.to_csv(csv_path)
    print(f"Saved model comparison: {csv_path}")
    return comparison


def print_key_findings(df: pd.DataFrame):
    """Print key findings for quick reference."""
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    for lang in df["language"].unique():
        lang_df = df[df["language"] == lang]
        lang_name = {"hau": "Hausa", "fon": "Fongbe"}.get(lang, lang)
        print(f"\n--- {lang_name} ---")
        print(f"  Total outputs: {len(lang_df)}")
        print(f"  Valid outputs: {lang_df['is_valid'].sum()} ({100*lang_df['is_valid'].mean():.1f}%)")
        print(f"  Avg word count: {lang_df['word_count'].mean():.1f}")
        print(f"  Avg TTR: {lang_df['ttr'].mean():.3f}")
        print(f"  Avg code-switching rate: {lang_df['code_switching_rate'].mean():.3f}")

        # Best task by TTR
        best_task = lang_df.groupby("task_type")["ttr"].mean().idxmax()
        print(f"  Best task for diversity (TTR): {best_task}")

        # Lowest code-switching task
        best_cs = lang_df.groupby("task_type")["code_switching_rate"].mean().idxmin()
        print(f"  Lowest code-switching: {best_cs}")


def main():
    config = load_config("configs/config.yaml")
    eval_dir = config["paths"]["eval_output_dir"]
    results_dir = config["paths"]["results_dir"]
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    print("Loading evaluations...")
    df = load_all_evaluations(eval_dir)

    if df.empty:
        print("No evaluations found. Run generation and evaluation first.")
        return

    print(f"Loaded {len(df)} evaluations")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Languages: {df['language'].unique().tolist()}")
    print(f"Tasks: {df['task_type'].unique().tolist()}")

    generate_summary_table(df, results_dir)
    generate_language_comparison(df, results_dir)
    generate_model_comparison(df, results_dir)
    print_key_findings(df)


if __name__ == "__main__":
    main()
