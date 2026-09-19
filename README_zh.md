# ⚡ Fast Agent Loop

> **[English](README.md) | 中文**

> **超轻量、生产级的 OpenAI 兼容 Agent 网关。**
> 纯 Python，零臃肿。原生 `/v1/chat/completions` SSE 流式输出 + 工具执行。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)

---

## 💡 为什么做 Fast Agent Loop？

大多数 Agent 运行时（LangChain、AutoGPT、CrewAI 或各种重型自研守护进程）都有企业落地的通病：
- **过重且慢**：庞大的依赖树、复杂的抽象层、缓慢的冷启动。
- **前端接入困难**：难以对接期待原生 OpenAI 流式协议（`text/event-stream`）的标准聊天 UI（NextChat、Open-WebUI、Streamlit、LibreChat 等）。
- **生产环境踩坑**：LLM 循环达到最大轮次时直接抛 500；工具输出过大把上下文撑爆；并发会话状态互相污染。

**Fast Agent Loop** 是一个单文件、约 400 行的异步 Agent 网关，能把任意上游 LLM（OpenAI、DeepSeek、Kimi、经适配的 Anthropic 等）与任意本地/远程工具桥接起来，提供磐石般的稳定性。

---

## 🚀 核心特性

- **标准 OpenAI 流式协议**：完整兼容 `/v1/chat/completions` SSE。直接向客户端流式输出 `delta.content`，并在工具调用轮次之间抑制内部 `[DONE]` 标记。
- **零框架臃肿**：仅基于 **FastAPI**、**Uvicorn** 和 **HTTPX** 构建。
- **安全的上下文压缩**：通过紧凑的 JSON 序列化保护上游 token 上限。工具输出超过 32KB 时自动以结构化提示拒绝，而不是粗暴截断数字或字符串。
- **平滑的轮次耗尽**：达到 `MAX_ROUNDS` 永远不会导致 HTTP 500 崩溃。Agent 会自动禁用工具调用，优雅地进入「尽力总结」轮。
- **并发与会话安全**：
  - **连接池**：模块级共享 `httpx.AsyncClient`，消除 TLS 握手延迟（每轮省约 200ms）。
  - **工具信号量**：限制并行工具执行数量（默认最多 4 路并发），保护下游服务。
  - **细粒度会话锁**：每个会话独立的异步锁，防止并发脏读与 prompt 竞态。

---

## 🛠️ 快速开始

### 1. 安装

```bash
git clone https://github.com/wangkun5212-hue/fast-agent-loop.git
cd fast-agent-loop
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export MODEL_BASE_URL="https://api.openai.com/v1" # 或 DeepSeek、Moonshot、SiliconFlow 等
export MODEL_API_KEY="sk-your-model-api-key"
export MODEL_NAME="gpt-4o-mini"
```

### 3. 注册工具并启动

创建 `main.py`：

```python
import uvicorn
from fast_agent_loop.server import app, tool_registry

# 定义你的业务工具
def get_weather(location: str):
    return {"location": location, "temp": 24, "condition": "Sunny"}

# 注册 schema 与处理函数
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

启动网关：
```bash
python main.py
```

---

## 🔌 对接任意聊天客户端

Fast Agent Loop 完全兼容 OpenAI 协议，可直接使用标准 OpenAI SDK 或 Web UI 接入：

### Python 客户端示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none"  # 如果配置了 GATEWAY_API_KEY 则填该值
)

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "东京今天天气怎么样？"}],
    stream=True,
    extra_headers={"X-Session-Id": "user_session_123"}
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## ⚙️ 配置项一览

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `MODEL_BASE_URL` | `https://api.openai.com/v1` | 上游 OpenAI 兼容 LLM 端点 |
| `MODEL_API_KEY` | *（必填）* | 上游模型 API Key |
| `MODEL_NAME` | `gpt-4o-mini` | 上游模型标识 |
| `GATEWAY_API_KEY` | `""`（开放） | 可选，要求网关客户端携带的 Bearer Token |
| `PORT` | `8000` | 网关 HTTP 监听端口 |
| `AGENT_MAX_ROUNDS` | `10` | 强制总结前的最大连续工具调用轮次 |
| `AGENT_TOOL_CONCURRENCY` | `4` | 工具并行执行的最大并发数 |
| `AGENT_TOOL_TIMEOUT` | `30.0` | 单次工具调用超时时间（秒） |
| `AGENT_MAX_SESSIONS` | `500` | 内存中 LRU 会话历史缓存大小 |

---

## 🐳 Docker 部署

```bash
docker build -t fast-agent-loop .
docker run -d -p 8000:8000 \
  -e MODEL_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_API_KEY="sk-xxxx" \
  -e MODEL_NAME="gpt-4o-mini" \
  fast-agent-loop
```

---

## 🧪 测试

运行测试套件：
```bash
pytest tests/ -v
```

---

## 📄 许可证

MIT License。商业或个人项目随意使用！
