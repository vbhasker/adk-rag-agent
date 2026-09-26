"""Run an ADK agent programmatically and capture a readable trace (tool calls, results, answer)."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def has_llm_credentials() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")) or os.getenv(
        "GOOGLE_GENAI_USE_VERTEXAI", ""
    ).lower() in {"1", "true"}


async def ask_async(agent: BaseAgent, question: str) -> dict[str, Any]:
    runner = Runner(agent=agent, app_name="rag_weekend", session_service=InMemorySessionService(), auto_create_session=True)
    message = types.Content(role="user", parts=[types.Part(text=question)])
    trace: list[dict[str, Any]] = []
    answer = ""
    try:
        async for event in runner.run_async(user_id="learner", session_id=str(uuid.uuid4()), new_message=message):
            for call in event.get_function_calls():
                trace.append({"type": "tool_call", "author": event.author, "name": call.name, "args": dict(call.args or {})})
            for response in event.get_function_responses():
                trace.append({"type": "tool_result", "author": event.author, "name": response.name, "response": response.response})
            if event.content and event.content.parts and not event.get_function_calls() and not event.get_function_responses():
                text = "".join(p.text or "" for p in event.content.parts if not getattr(p, "thought", False)).strip()
                if text:
                    trace.append({"type": "message", "author": event.author, "text": text})
                    if event.is_final_response():
                        answer = text
    finally:
        await runner.close()
    return {"answer": answer, "trace": trace}


def ask(agent: BaseAgent, question: str) -> dict[str, Any]:
    return asyncio.run(ask_async(agent, question))
