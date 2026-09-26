"""Small helpers for calling an LLM directly (outside an agent): Gemini by default, OpenAI optional.

Used for "LLM as a component" jobs: reranking here, query rewriting in Chapter 6, judging in Chapter 7.
Structured output (a Pydantic schema) means we get validated JSON back instead of parsing prose.

Provider and model come from llm_config.py (LLM_PROVIDER, GEMINI_MODEL / OPENAI_MODEL).
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from llm_config import has_llm_credentials, model_name, provider  # noqa: F401  (re-exported)

T = TypeVar("T", bound=BaseModel)

_CLIENTS: dict[str, object] = {}


def client():
    """The SDK client for the active provider (created once)."""
    name = provider()
    if name not in _CLIENTS:
        if name == "openai":
            from openai import OpenAI

            _CLIENTS[name] = OpenAI()  # reads OPENAI_API_KEY (and OPENAI_BASE_URL for compatible endpoints)
        else:
            from google import genai

            _CLIENTS[name] = genai.Client()  # GOOGLE_API_KEY, or GOOGLE_GENAI_USE_VERTEXAI + project/location
    return _CLIENTS[name]


def generate_json(prompt: str, schema: type[T], temperature: float = 0.0) -> T:
    if provider() == "openai":
        # No temperature: OpenAI reasoning models (gpt-5 family, o-series) only accept the default.
        response = client().responses.parse(model=model_name(), input=prompt, text_format=schema)
        return response.output_parsed

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
    if provider() == "openai":
        return (client().responses.create(model=model_name(), input=prompt).output_text or "").strip()

    from google.genai import types

    response = client().models.generate_content(
        model=model_name(), contents=prompt, config=types.GenerateContentConfig(temperature=temperature)
    )
    return (response.text or "").strip()
