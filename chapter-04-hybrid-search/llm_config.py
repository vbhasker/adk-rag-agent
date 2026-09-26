"""Which LLM powers the agents: Gemini (default) or OpenAI.

    LLM_PROVIDER=gemini   → GEMINI_MODEL (e.g. gemini-2.5-flash), via GOOGLE_API_KEY or Vertex AI
    LLM_PROVIDER=openai   → OPENAI_MODEL (e.g. gpt-5-mini), via OPENAI_API_KEY

ADK talks to Gemini natively; any other provider goes through ADK's LiteLlm wrapper
(model string "openai/<name>"), so the agents themselves don't change at all.
"""

from __future__ import annotations

import os


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").lower()


def has_google_credentials() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")) or os.getenv(
        "GOOGLE_GENAI_USE_VERTEXAI", ""
    ).lower() in {"1", "true"}


def has_openai_credentials() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def has_llm_credentials() -> bool:
    return has_openai_credentials() if provider() == "openai" else has_google_credentials()


def model_name() -> str:
    if provider() == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-5-mini")
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def agent_model():
    """The `model=` value for an ADK Agent: a plain string for Gemini, a LiteLlm object for OpenAI."""
    if provider() == "openai":
        from google.adk.models.lite_llm import LiteLlm  # needs `pip install litellm`

        return LiteLlm(model=f"openai/{model_name()}")
    return model_name()


MISSING_CREDENTIALS = (
    "No LLM credentials. Add GOOGLE_API_KEY (or Vertex AI settings) to .env, "
    "or set LLM_PROVIDER=openai with OPENAI_API_KEY, then restart."
)
