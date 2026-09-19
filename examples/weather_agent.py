#!/usr/bin/env python3
"""
Example: Running a Weather & Search Agent with Fast Agent Loop
"""

import asyncio
from fast_agent_loop.server import tool_registry, app
import uvicorn

# 1. Define Tool Handlers
def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """Mock weather service"""
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "condition": "Sunny",
        "unit": unit
    }

async def search_web(query: str) -> dict:
    """Mock search engine"""
    await asyncio.sleep(0.1)  # simulate async network call
    return {
        "query": query,
        "results": [
            {"title": "Fast Agent Loop Release", "snippet": "A production-grade OpenAI-compatible gateway."}
        ]
    }

# 2. Register OpenAI Tool Schemas
tool_registry.register(
    schema={
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    },
    handler=get_current_weather
)

tool_registry.register(
    schema={
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for recent events and news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"}
                },
                "required": ["query"]
            }
        }
    },
    handler=search_web
)

if __name__ == "__main__":
    print("Starting Fast Agent Loop with 2 tools registered...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
