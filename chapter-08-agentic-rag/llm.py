"""Small helpers for calling Gemini directly (outside an agent) via the google-genai SDK.

Used for "LLM as a component" jobs: reranking here, query rewriting in Chapter 6, judging in Chapter 7.
Structured output (a Pydantic schema) means we get validated JSON back instead of parsing prose.
"""

from __future__ import annotations

import os
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_CLIENT = None


def has_llm_credentials() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")) or os.getenv(
        "GOOGLE_GENAI_USE_VERTEXAI", ""
    ).lower() in {"1", "true"}


def model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai

        _CLIENT = genai.Client()  # GOOGLE_API_KEY, or GOOGLE_GENAI_USE_VERTEXAI + project/location
    return _CLIENT


def generate_json(prompt: str, schema: type[T], temperature: float = 0.0) -> T:
    from google.genai import types

    response = client().models.generate_content(
        model=model_name(),
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature, response_mime_type="application/json", response_schema=schema
        ),
    )
    return schema.model_validate_json(response.text)


def generate_text(prompt: str, temperature: float = 0.2) -> str:
    from google.genai import types

    response = client().models.generate_content(
        model=model_name(), contents=prompt, config=types.GenerateContentConfig(temperature=temperature)
    )
    return (response.text or "").strip()
