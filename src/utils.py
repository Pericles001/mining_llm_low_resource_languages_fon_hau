"""
Utility functions: config loading, file I/O, logging.
"""

import yaml
import os


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_all_dirs(config: dict):
    """Create all required directories from config."""
    dirs = [
        config["paths"]["prompts_dir"],
        config["paths"]["raw_output_dir"],
        config["paths"]["eval_output_dir"],
        config["paths"]["results_dir"],
        config["paths"]["human_eval_dir"],
        os.path.join(config["paths"]["results_dir"], "tables"),
        os.path.join(config["paths"]["results_dir"], "figures"),
    ]
    # Raw output subdirs
    for model in config["models"]:
        for lang in config["languages"]:
            lang_folder = {"hau": "hausa", "fon": "fongbe"}[lang["code"]]
            for task in config["task_types"]:
                dirs.append(
                    os.path.join(
                        config["paths"]["raw_output_dir"], model, lang_folder, task
                    )
                )
                dirs.append(
                    os.path.join(
                        config["paths"]["eval_output_dir"], model, lang_folder, task
                    )
                )

    for d in dirs:
        os.makedirs(d, exist_ok=True)
