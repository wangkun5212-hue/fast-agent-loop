#!/usr/bin/env python3
"""
Fast Agent Loop - Lightweight OpenAI-Compatible Agent Gateway

Features:
  - Native OpenAI /v1/chat/completions SSE streaming with function calling
  - Zero-dependency runtime (only FastAPI, Uvicorn, HTTPX)
  - Shared HTTP connection pool for minimal latency
  - Bounded tool execution concurrency (semaphore protected)
  - Safe context compression (refuses rather than corrupting large JSON)
  - Session serialization lock (prevents concurrent dirty writes)
  - Smooth round exhaustion fallback (never crash with 500 on round limits)
"""

import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ── Environment Configuration ───────────────────────────────
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "8000"))

MAX_ROUNDS = int(os.getenv("AGENT_MAX_ROUNDS", "10"))
MAX_SESSIONS = int(os.getenv("AGENT_MAX_SESSIONS", "500"))
MAX_HISTORY = int(os.getenv("AGENT_MAX_HISTORY", "10"))
TOOL_TIMEOUT = float(os.getenv("AGENT_TOOL_TIMEOUT", "30.0"))
TOOL_CONCURRENCY = int(os.getenv("AGENT_TOOL_CONCURRENCY", "4"))

DEFAULT_SYSTEM_PROMPT = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    "You are a helpful and reliable AI Agent. "
    "Always base your answers strictly on verified tool results. "
    "Do not invent facts, numbers, or unverified claims."
)

# ── Concurrency & Session State ─────────────────────────────
_TOOL_SEM = asyncio.Semaphore(TOOL_CONCURRENCY)
_session_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
_session_histories: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()


def _get_session_lock(session_id: str) -> asyncio.Lock:
    lock = _session_locks.get(session_id)
    if lock is not None:
        _session_locks.move_to_end(session_id)
        return lock

    # Never evict a held lock: replacing it would allow two requests for the
    # same session to execute concurrently. If every lock is busy, temporary
    # overflow is safer and is pruned when streams complete.
    if len(_session_locks) >= MAX_SESSIONS:
        for key, candidate in list(_session_locks.items()):
            if not candidate.locked():
                del _session_locks[key]
                break

    lock = asyncio.Lock()
    _session_locks[session_id] = lock
    return lock


def _prune_session_locks() -> None:
    while len(_session_locks) > MAX_SESSIONS:
        removed = False
        for key, lock in list(_session_locks.items()):
            if not lock.locked():
                del _session_locks[key]
                removed = True
                break
        if not removed:
            break


def _get_session_history(session_id: str) -> List[Dict[str, Any]]:
    if session_id in _session_histories:
        _session_histories.move_to_end(session_id)
        return list(_session_histories[session_id])
    return []


def _save_session_history(session_id: str, history: List[Dict[str, Any]]) -> None:
    trimmed = history[-MAX_HISTORY:]
    _session_histories[session_id] = trimmed
    _session_histories.move_to_end(session_id)
    while len(_session_histories) > MAX_SESSIONS:
        _session_histories.popitem(last=False)


# ── Tool Registry ───────────────────────────────────────────
class ToolRegistry:
    def __init__(self):
        self.schemas: List[Dict[str, Any]] = []
        self.handlers: Dict[str, Callable] = {}

    def register(self, schema: Dict[str, Any], handler: Callable):
        tool_name = schema.get("function", {}).get("name")
        if not tool_name:
            raise ValueError("Invalid schema: function.name is required")
        self.schemas = [
            existing
            for existing in self.schemas
            if existing.get("function", {}).get("name") != tool_name
        ]
        self.schemas.append(schema)
        self.handlers[tool_name] = handler

    def get_schemas(self) -> List[Dict[str, Any]]:
        return self.schemas

    async def execute(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self.handlers:
            return json.dumps({"error": f"Tool '{name}' not found"})
        handler = self.handlers[name]
        async with _TOOL_SEM:
            try:
                if asyncio.iscoroutinefunction(handler):
                    res = await asyncio.wait_for(handler(**args), timeout=TOOL_TIMEOUT)
                else:
                    res = await asyncio.wait_for(
                        asyncio.to_thread(handler, **args), timeout=TOOL_TIMEOUT
                    )
                if isinstance(res, str):
                    return res
                return json.dumps(res, ensure_ascii=False)
            except asyncio.TimeoutError:
                return json.dumps({"error": f"Tool '{name}' execution timed out ({TOOL_TIMEOUT}s)"})
            except Exception as e:
                return json.dumps({"error": f"Tool '{name}' failed: {str(e)}"})


tool_registry = ToolRegistry()


# ── Tool Context Guard ──────────────────────────────────────
def _compress_tool_context(result: str, max_chars: int = 32000) -> str:
    """
    Compress tool JSON safely.
    If exceeding limit, return a structured notice instead of truncating numbers/JSON.
    """
    try:
        compact = json.dumps(json.loads(result), ensure_ascii=False, separators=(",", ":"))
    except (ValueError, TypeError):
        compact = result

    if len(compact) <= max_chars:
        return compact

    return json.dumps(
        {
            "error": "Tool output is too large. Truncation avoided to prevent data corruption. "
                     "Please refine your query or specify filtering parameters.",
            "length": len(compact),
        },
        ensure_ascii=False,
    )


# ── Client & Lifecycle ──────────────────────────────────────
_shared_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _shared_client
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=16)
    _shared_client = httpx.AsyncClient(limits=limits, timeout=60.0)
    yield
    if _shared_client:
        await _shared_client.aclose()


app = FastAPI(title="Fast Agent Loop", version="1.0.0", lifespan=lifespan)


# ── Core Agent Loop ─────────────────────────────────────────
async def _agent_stream_generator(
    messages: List[Dict[str, Any]],
    session_id: str,
    tools_schema: List[Dict[str, Any]],
    system_prompt: str,
) -> AsyncGenerator[str, None]:
    client = _shared_client or httpx.AsyncClient()
    created = int(time.time())

    # Prepare conversation history
    conversation = [{"role": "system", "content": system_prompt}]
    if session_id:
        conversation.extend(_get_session_history(session_id))

    for m in messages:
        if m.get("role") != "system":
            conversation.append(m)

    final_assistant_content = ""
    answered = False

    for round_idx in range(MAX_ROUNDS):
        is_final_round = (round_idx == MAX_ROUNDS - 1)
        if is_final_round:
            conversation.append({
                "role": "system",
                "content": "Maximum tool rounds reached. Do not call more tools. "
                           "Summarize the best supported conclusion from the results so far.",
            })
        payload: Dict[str, Any] = {
            "model": MODEL_NAME,
            "messages": conversation,
            "stream": True,
        }

        if tools_schema and not is_final_round:
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {MODEL_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            req = client.build_request("POST", f"{MODEL_BASE_URL}/chat/completions", json=payload, headers=headers)
            resp = await client.send(req, stream=True)
            resp.raise_for_status()
        except Exception as e:
            err_msg = f"Upstream LLM error: {str(e)}"
            yield f"data: {json.dumps({'choices': [{'delta': {'content': err_msg}}]}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        current_tool_calls: Dict[int, Dict[str, Any]] = {}
        round_content = ""

        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                continue

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            # Stream text content
            if delta.get("content"):
                round_content += delta["content"]
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # Accumulate tool calls
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        current_tool_calls[idx]["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        current_tool_calls[idx]["function"]["arguments"] += fn["arguments"]

        await resp.aclose()

        # Case A: Model returned text response
        if round_content and not current_tool_calls:
            final_assistant_content = round_content
            answered = True
            break

        # Case B: Model called tools
        if current_tool_calls and not is_final_round:
            tool_calls_list = list(current_tool_calls.values())
            conversation.append({
                "role": "assistant",
                "content": round_content or None,
                "tool_calls": tool_calls_list,
            })

            # Parallel tool execution
            async def _run_tool(call_item: Dict[str, Any]):
                call_id = call_item["id"]
                fn_name = call_item["function"]["name"]
                raw_args = call_item["function"]["arguments"]
                try:
                    parsed_args = json.loads(raw_args) if raw_args else {}
                except Exception:
                    parsed_args = {}
                result_str = await tool_registry.execute(fn_name, parsed_args)
                safe_content = _compress_tool_context(result_str)
                return {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": safe_content,
                }

            tool_results = await asyncio.gather(*[_run_tool(c) for c in tool_calls_list])
            conversation.extend(tool_results)
            continue

        # Case C: Round limit reached
        if is_final_round:
            final_assistant_content = round_content
            answered = True
            break

    # Save session history if answered
    if session_id and answered:
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if last_user:
            hist = _get_session_history(session_id)
            hist.append(last_user)
            hist.append({"role": "assistant", "content": final_assistant_content})
            _save_session_history(session_id, hist)

    yield "data: [DONE]\n\n"


# ── Routes ──────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "tools_registered": len(tool_registry.get_schemas()),
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str = Header(default=""),
    x_session_id: str = Header(default="", alias="X-Session-Id"),
    x_hermes_session_id: str = Header(default="", alias="X-Hermes-Session-Id"),
):
    session_id = x_session_id or x_hermes_session_id

    if GATEWAY_API_KEY:
        if authorization != f"Bearer {GATEWAY_API_KEY}":
            raise HTTPException(status_code=401, detail="Invalid API Key")

    if not MODEL_API_KEY:
        raise HTTPException(status_code=503, detail="MODEL_API_KEY is not configured")

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")

    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")

    requested_tools = body.get("tools")
    if requested_tools is not None and not isinstance(requested_tools, list):
        raise HTTPException(status_code=400, detail="'tools' must be a list")

    registered_tools = {
        schema.get("function", {}).get("name"): schema
        for schema in tool_registry.get_schemas()
    }
    if requested_tools is not None:
        requested_names = []
        for schema in requested_tools:
            if not isinstance(schema, dict):
                raise HTTPException(status_code=400, detail="Each tool must be an object")
            name = schema.get("function", {}).get("name")
            if not name:
                raise HTTPException(
                    status_code=400,
                    detail="Each tool must define function.name",
                )
            requested_names.append(name)
        unknown_names = [name for name in requested_names if name not in registered_tools]
        if unknown_names:
            raise HTTPException(
                status_code=400,
                detail=f"Unregistered tools requested: {', '.join(str(n) for n in unknown_names)}",
            )
        tools = [registered_tools[name] for name in requested_names]
    else:
        tools = list(registered_tools.values())
    system_prompt = body.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    # Stream response
    if body.get("stream", True):
        async def locked_stream():
            if session_id:
                lock = _get_session_lock(session_id)
                try:
                    async with lock:
                        async for chunk in _agent_stream_generator(messages, session_id, tools, system_prompt):
                            yield chunk
                finally:
                    _prune_session_locks()
            else:
                async for chunk in _agent_stream_generator(messages, session_id, tools, system_prompt):
                    yield chunk

        return StreamingResponse(
            locked_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming fallback
    async def collect_response() -> str:
        collected = ""
        async for chunk in _agent_stream_generator(messages, session_id, tools, system_prompt):
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                try:
                    obj = json.loads(chunk[6:])
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        collected += delta["content"]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
        return collected

    if session_id:
        lock = _get_session_lock(session_id)
        try:
            async with lock:
                collected_text = await collect_response()
        finally:
            _prune_session_locks()
    else:
        collected_text = await collect_response()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": collected_text},
                "finish_reason": "stop",
            }
        ],
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[fast_agent_loop] Unhandled Exception: {request.method} {request.url.path}", flush=True)
    traceback.print_exc(file=sys.stdout)
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
