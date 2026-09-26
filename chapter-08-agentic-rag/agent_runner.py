"""Run an ADK agent (single or multi-agent) programmatically and capture a readable trace.

For multi-agent workflows the "answer" is not simply the last message (the fact checker may speak last),
so we also return the final session state, where each agent stored its output_key.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from llm_config import has_llm_credentials  # noqa: F401  (re-exported for server.py / ask.py)


async def ask_async(agent: BaseAgent, question: str, answer_key: str | None = None) -> dict[str, Any]:
    """answer_key: state key holding the final answer (e.g. "draft_answer"); default = last final message."""
    runner = Runner(agent=agent, app_name="rag_weekend", session_service=InMemorySessionService(), auto_create_session=True)
    session_id = str(uuid.uuid4())
    message = types.Content(role="user", parts=[types.Part(text=question)])
    trace: list[dict[str, Any]] = []
    answer = ""
    start = time.perf_counter()
    try:
        async for event in runner.run_async(user_id="learner", session_id=session_id, new_message=message):
            elapsed = round(time.perf_counter() - start, 2)
            for call in event.get_function_calls():
                trace.append({"type": "tool_call", "author": event.author, "name": call.name, "args": dict(call.args or {}), "t": elapsed})
            for response in event.get_function_responses():
                trace.append({"type": "tool_result", "author": event.author, "name": response.name, "response": response.response, "t": elapsed})
            if event.content and event.content.parts and not event.get_function_calls() and not event.get_function_responses():
                text = "".join(p.text or "" for p in event.content.parts if not getattr(p, "thought", False)).strip()
                if text:
                    trace.append({"type": "message", "author": event.author, "text": text, "t": elapsed})
                    if event.is_final_response():
                        answer = text
        session = await runner.session_service.get_session(app_name="rag_weekend", user_id="learner", session_id=session_id)
        state = dict(session.state) if session else {}
    finally:
        await runner.close()
    if answer_key and state.get(answer_key):
        answer = state[answer_key]
    return {"answer": answer, "trace": trace, "state": state, "seconds": round(time.perf_counter() - start, 2)}


def ask(agent: BaseAgent, question: str, answer_key: str | None = None) -> dict[str, Any]:
    return asyncio.run(ask_async(agent, question, answer_key))
