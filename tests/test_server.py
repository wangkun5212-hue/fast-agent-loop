#!/usr/bin/env python3
import asyncio
import json
from collections import OrderedDict

import pytest

import fast_agent_loop.server as server
from fast_agent_loop.server import (
    ToolRegistry,
    _compress_tool_context,
    _get_session_history,
    _save_session_history,
    tool_registry,
)

def test_tool_context_compression():
    small_data = json.dumps({"status": "ok", "items": [1, 2, 3]})
    assert _compress_tool_context(small_data, max_chars=100) == '{"status":"ok","items":[1,2,3]}'

    # Exceed limit
    huge_data = json.dumps({"data": "x" * 500})
    res = _compress_tool_context(huge_data, max_chars=100)
    parsed = json.loads(res)
    assert "error" in parsed
    assert "Tool output is too large" in parsed["error"]

def test_tool_registry():
    def add(a: int, b: int) -> int:
        return a + b

    tool_registry.register(
        schema={
            "type": "function",
            "function": {
                "name": "calc_add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }
            }
        },
        handler=add
    )

    schemas = tool_registry.get_schemas()
    assert any(s["function"]["name"] == "calc_add" for s in schemas)


def test_reregistering_tool_replaces_schema_and_handler():
    registry = ToolRegistry()
    schema = {
        "type": "function",
        "function": {"name": "same_name", "parameters": {"type": "object"}},
    }
    registry.register(schema=schema, handler=lambda: "first")
    registry.register(schema=schema, handler=lambda: "second")

    assert len(registry.get_schemas()) == 1
    assert asyncio.run(registry.execute("same_name", {})) == "second"


def test_busy_session_lock_is_not_evicted(monkeypatch):
    monkeypatch.setattr(server, "MAX_SESSIONS", 1)
    monkeypatch.setattr(server, "_session_locks", OrderedDict())

    first = server._get_session_lock("first")

    async def exercise_locks():
        await first.acquire()
        try:
            second = server._get_session_lock("second")
            assert server._session_locks["first"] is first
            assert second is server._session_locks["second"]
        finally:
            first.release()
        server._prune_session_locks()

    asyncio.run(exercise_locks())
    assert len(server._session_locks) == 1

def test_session_history_lru():
    _save_session_history("sess_1", [{"role": "user", "content": "hi"}])
    hist = _get_session_history("sess_1")
    assert len(hist) == 1
    assert hist[0]["content"] == "hi"
