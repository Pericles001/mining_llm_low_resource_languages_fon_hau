# Mining LLMs for Low-Resource Language Data

## Research Goal
Systematically extract usable text data from LLMs for Hausa and Fongbe by comparing 6 elicitation task types across 2 models (GPT-4o Mini, Gemini 2.5 Flash).

## Project Structure

```
mining_llm_project/
│
├── configs/
│   └── config.yaml              # API keys, model settings, generation params
│
├── prompts/
│   ├── creative_writing.json     # Poems, stories, folktales, songs
│   ├── functional_text.json      # Letters, instructions, news, recipes
│   ├── structured_knowledge.json # Definitions, proverbs, translations
│   ├── dialogue.json             # Conversations, interviews, negotiations
│   ├── topic_switching.json      # Start topic A, switch to topic B
│   └── constrained_generation.json # Write using specific words / no code-switching
│
├── src/
│   ├── __init__.py
│   ├── generator.py              # Core: sends prompts to LLMs, collects outputs
│   ├── models.py                 # Model wrappers for Gemini and GPT-4o Mini
│   ├── prompt_loader.py          # Loads and formats prompts per language/task
│   ├── evaluator.py              # Automatic evaluation (diversity, code-switching, etc.)
│   ├── language_detector.py      # Detects language fidelity using fastText/langdetect
│   └── utils.py                  # Helpers: file I/O, logging, token counting
│
├── scripts/
│   ├── run_generation.py         # CLI entry point: generate all outputs
│   ├── run_evaluation.py         # CLI entry point: evaluate all outputs
│   └── run_analysis.py           # CLI entry point: aggregate results, tables, plots
│
├── outputs/
│   ├── raw/                      # Raw LLM outputs organized by model/lang/task
│   │   ├── gemini/
│   │   │   ├── hausa/
│   │   │   │   ├── creative_writing/
│   │   │   │   ├── functional_text/
│   │   │   │   ├── structured_knowledge/
│   │   │   │   ├── dialogue/
│   │   │   │   ├── topic_switching/
│   │   │   │   └── constrained_generation/
│   │   │   └── fongbe/
│   │   │       └── ... (same 6 folders)
│   │   └── gpt4o_mini/
│   │       ├── hausa/
│   │       │   └── ... (same 6 folders)
│   │       └── fongbe/
│   │           └── ... (same 6 folders)
│   │
│   └── evaluated/                # Evaluation scores per output
│       ├── gemini/
│       └── gpt4o_mini/
│
├── results/
│   ├── tables/                   # LaTeX tables for paper
│   ├── figures/                  # Plots
│   └── summary.csv              # Aggregated results
│
├── human_eval/
│   ├── templates/                # Survey templates for native speakers
│   └── responses/                # Collected human evaluation data
│
├── requirements.txt
├── .env                          # API keys (not committed)
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys in .env
cp .env.example .env
# Edit .env with your keys

# 3. Generate outputs (all models, all languages, all tasks)
python scripts/run_generation.py --all

# 4. Run automatic evaluation
python scripts/run_evaluation.py --all

# 5. Generate analysis tables and figures
python scripts/run_analysis.py
```

## Elicitation Task Types (6 categories, 25 prompts each = 150 total per language)

1. **Creative Writing** (25 prompts): poems, short stories, folktales, songs, proverbs
2. **Functional Text** (25 prompts): letters, instructions, news articles, recipes, announcements
3. **Structured Knowledge** (25 prompts): definitions, word explanations, grammar examples, translations
4. **Dialogue** (25 prompts): conversations, interviews, market negotiations, family discussions
5. **Topic Switching** (25 prompts): start on topic A, mid-prompt switch to topic B
6. **Constrained Generation** (25 prompts): use specific vocabulary, avoid code-switching, use only target language

## Evaluation Dimensions (4 automatic + human)

1. **Linguistic Accuracy**: language ID confidence, diacritic correctness (Fongbe)
2. **Lexical Diversity**: TTR, unique vocabulary, hapax legomena ratio
3. **Domain Coverage**: topic distribution across outputs
4. **Code-Switching Rate**: proportion of non-target language tokens
5. **Human Evaluation** (separate): native speaker 1-5 scores on fluency, accuracy, naturalness
