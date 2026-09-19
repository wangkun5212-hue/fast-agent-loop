#!/usr/bin/env python3
import pytest
import json
from fast_agent_loop.server import _compress_tool_context, tool_registry, _save_session_history, _get_session_history

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

def test_session_history_lru():
    _save_session_history("sess_1", [{"role": "user", "content": "hi"}])
    hist = _get_session_history("sess_1")
    assert len(hist) == 1
    assert hist[0]["content"] == "hi"
