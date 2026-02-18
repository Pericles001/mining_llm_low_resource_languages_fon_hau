"""
Model wrappers for Gemini 2.5 Flash and GPT-4o Mini.
Each wrapper provides a unified generate() interface.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()


class GeminiModel:
    """Wrapper for Google Gemini 2.5 Flash."""

    def __init__(self, config: dict):
        from google import genai

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = config.get("name", "gemini-2.5-flash")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_output_tokens", 1024)
        self.top_p = config.get("top_p", 0.95)

    def generate(self, prompt: str) -> dict:
        """Generate text from a prompt. Returns dict with text, tokens, metadata."""
        from google.genai import types

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=self.top_p,
                ),
            )
            text = response.text if response.text else ""
            return {
                "text": text,
                "model": self.model_name,
                "provider": "google",
                "success": True,
                "error": None,
                "token_count": len(text.split()),  # approximate
            }
        except Exception as e:
            return {
                "text": "",
                "model": self.model_name,
                "provider": "google",
                "success": False,
                "error": str(e),
                "token_count": 0,
            }


class GPT4oMiniModel:
    """Wrapper for OpenAI GPT-4o Mini."""

    def __init__(self, config: dict):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = config.get("name", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)
        self.top_p = config.get("top_p", 0.95)

    def generate(self, prompt: str) -> dict:
        """Generate text from a prompt. Returns dict with text, tokens, metadata."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates text in African languages. Always respond in the requested language only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            text = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
            return {
                "text": text,
                "model": self.model_name,
                "provider": "openai",
                "success": True,
                "error": None,
                "token_count": tokens_used,
            }
        except Exception as e:
            return {
                "text": "",
                "model": self.model_name,
                "provider": "openai",
                "success": False,
                "error": str(e),
                "token_count": 0,
            }


def get_model(model_key: str, config: dict):
    """Factory function to get a model by key."""
    if model_key == "gemini":
        return GeminiModel(config["models"]["gemini"])
    elif model_key == "gpt4o_mini":
        return GPT4oMiniModel(config["models"]["gpt4o_mini"])
    else:
        raise ValueError(f"Unknown model: {model_key}")
