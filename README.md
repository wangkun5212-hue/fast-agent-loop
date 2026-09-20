# ⚡ Fast Agent Loop

> **A lightweight OpenAI-compatible Agent Gateway for local tool execution.**
> Pure Python. Zero bloat. Native `/v1/chat/completions` SSE streaming with tool execution.

> **English | [中文文档](README_zh.md)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)

---

## 💡 Why Fast Agent Loop?

Most agent runtimes (LangChain, AutoGPT, CrewAI, or heavy custom daemons) suffer from common enterprise pain points:
- **Overkill & Slow**: Massive dependency trees, complex abstractions, and slow cold starts.
- **Frontend Friction**: Hard to connect to standard chat UIs (NextChat, Open-WebUI, Streamlit, LibreChat) that expect native OpenAI streaming (`text/event-stream`).
- **Production Pitfalls**: LLM loops often throw 500 errors when reaching max rounds, choke on huge tool outputs, or suffer from concurrent session state corruption.

**Fast Agent Loop** is a compact asynchronous agent gateway that bridges an OpenAI-compatible upstream LLM with registered local or remote tools.

---

## 🚀 Key Features

- **OpenAI-style Streaming**: Provides a `/v1/chat/completions` SSE endpoint, streams `delta.content`, and suppresses internal `[DONE]` tokens between tool rounds.
- **Zero Framework Bloat**: Built purely on top of **FastAPI**, **Uvicorn**, and **HTTPX**.
- **Safe Context Compression**: Protects upstream token limits by compactly serializing JSON outputs. Automatically rejects outputs >32KB with a structured notice instead of violently slicing numbers or strings.
- **Smooth Round Exhaustion**: Reaching `MAX_ROUNDS` never crashes with HTTP 500. The agent disables tool calling and enters a graceful "best-effort summary" turn.
- **Concurrency & Session Safety**:
  - **Connection Pool**: Module-level shared `httpx.AsyncClient` eliminates TLS handshake latency (~200ms saved per round).
  - **Tool Semaphore**: Enforces bounded parallel tool execution (default max 4 concurrent calls) to protect downstream services.
  - **Fine-grained Session Lock**: Asynchronous lock per session prevents concurrent dirty reads and prompt race conditions.

---

## 🛠️ Quick Start

### 1. Installation

```bash
git clone https://github.com/<your-username>/fast-agent-loop.git
cd fast-agent-loop
pip install -e .
```

### 2. Configure Environment

```bash
export MODEL_BASE_URL="https://api.openai.com/v1" # Or DeepSeek, Moonshot, SiliconFlow, etc.
export MODEL_API_KEY="sk-your-model-api-key"
export MODEL_NAME="gpt-4o-mini"
```

### 3. Register Tools & Run

Create `main.py`:

```python
import uvicorn
from fast_agent_loop.server import app, tool_registry

# Define your business tool
def get_weather(location: str):
    return {"location": location, "temp": 24, "condition": "Sunny"}

# Register schema and handler
tool_registry.register(
    schema={
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    },
    handler=get_weather
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Start the gateway:
```bash
python main.py
```

---

## 🔌 Connect Any Chat Client

Because Fast Agent Loop is fully OpenAI-compatible, you can connect with standard OpenAI SDKs or Web UIs:

### Python Client Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none"  # Or GATEWAY_API_KEY if configured
)

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    stream=True,
    extra_headers={"X-Session-Id": "user_session_123"}
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## ⚙️ Configuration Reference

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_BASE_URL` | `https://api.openai.com/v1` | Upstream OpenAI-compatible LLM endpoint |
| `MODEL_API_KEY` | *(Required)* | API key for upstream model |
| `MODEL_NAME` | `gpt-4o-mini` | Upstream model identifier |
| `GATEWAY_API_KEY` | `""` (Open) | Optional Bearer token required for gateway clients |
| `PORT` | `8000` | Gateway HTTP listen port |
| `AGENT_MAX_ROUNDS` | `10` | Maximum continuous tool rounds before forcing summary |
| `AGENT_MAX_HISTORY` | `10` | Maximum stored messages per stateful session |
| `AGENT_TOOL_CONCURRENCY` | `4` | Maximum parallel tool execution concurrency |
| `AGENT_TOOL_TIMEOUT` | `30.0` | Timeout in seconds for individual tool calls |
| `AGENT_MAX_SESSIONS` | `500` | In-memory LRU session history cache size |

`X-Session-Id` enables server-side history. In that mode, send only the new turn in
`messages`; clients that already send the complete conversation should omit the header.

---

## 🐳 Docker Deployment

```bash
docker build -t fast-agent-loop .
docker run -d -p 8000:8000 \
  -e MODEL_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_API_KEY="sk-xxxx" \
  -e MODEL_NAME="gpt-4o-mini" \
  fast-agent-loop
```

---

## 🧪 Testing

Run test suite:
```bash
pytest tests/ -v
```

---

## 📄 License

MIT License. Feel free to use in your commercial or personal projects!
