# Personal Agent

A personal AI agent built on [nanobot](https://github.com/HKUDS/nanobot), featuring an orchestrator + expert subagent architecture with persistent memory and semantic search.

## Architecture

```
User → Channel (CLI/Telegram/...) → MessageBus → Orchestrator → Expert Subagents
                                                    ↕                ↕
                                              Memory System     Expert Memory
                                              (MEMORY.md +      (MEMORY.md +
                                               vector DB)        SOUL.md +
                                                                 EXPERIENCE.md)
```

**Orchestrator** — Receives messages, maintains conversation context, delegates tasks to expert subagents via `spawn`.

**Expert Subagents** — Execute specific tasks with isolated workspaces, persistent memory, and full tool access (files, shell, web). Each expert is automatically profiled after its first run and reused for similar future tasks.

## Memory System

### Main Agent Memory

Three-tier architecture with semantic search:

| Tier | Storage | Retention | Purpose |
|------|---------|-----------|---------|
| Working | In-memory | Session | Current conversation context |
| Session | SQLite + vectors | 30 days | Recent conversations with semantic search |
| Long-term | MEMORY.md + vectors | Permanent | Consolidated facts and knowledge |

Embeddings use **Ollama** (`qwen3-embedding:0.6b`) locally. Falls back to basic keyword-based memory if Ollama is unavailable.

### Expert Subagent Memory

Each expert has its own persistent memory files in `experts/{name}/memory/`:

| File | Purpose |
|------|---------|
| `MEMORY.md` | What works, what doesn't — task-specific knowledge |
| `SOUL.md` | Identity and personality — consistent behavior across runs |
| `EXPERIENCE.md` | Lessons learned — auto-extracted by LLM after each task |

Experts load all memory files into their system prompt on each run, so they **remember previous experiences** and **improve over time** without retraining.

### Expert Directory Layout

```
experts/{name}/
├── EXPERT.md           # Auto-generated profile
├── WORKLOG.md          # Live work log during execution
├── memory/
│   ├── MEMORY.md       # Persistent knowledge
│   ├── HISTORY.md      # Grep-searchable task log
│   ├── SOUL.md         # Identity and personality
│   └── EXPERIENCE.md   # Lessons learned
├── workspace/          # Isolated file sandbox
├── sessions/           # Persistent conversation history
└── results/            # Timestamped result files
```

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) (for local embeddings)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install loguru litellm pydantic pydantic-settings numpy httpx chromadb whoosh json-repair typer prompt_toolkit rich tiktoken ddgs readability-lxml oauth-cli-kit
```

### Install Embedding Model

```bash
ollama pull qwen3-embedding:0.6b
```

### Configure

Edit `~/.nanobot/config.json`:

```json
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5",
      "workspace": "~/.nanobot/workspace"
    }
  },
  "providers": {
    "anthropic": {
      "api_key": "sk-ant-..."
    }
  }
}
```

Supports 20+ providers: Anthropic, OpenAI, DeepSeek, Gemini, Ollama, OpenRouter, etc. Provider auto-detection is based on model name keywords.

### Run

```bash
source .venv/bin/activate
python3 -m nanobot
```

## Tool System

**Orchestrator tools** (minimal, for delegation):
- `spawn` — Delegate task to an expert subagent
- `message` — Send message to user
- `cron` — Schedule tasks

**Expert tools** (full, for task execution):
- `read_file`, `write_file`, `edit_file`, `list_dir` — Filesystem (sandboxed to expert workspace)
- `exec` — Shell commands (restricted to workspace)
- `web_search`, `web_fetch` — Web search and content retrieval
- MCP tools — External tool servers via Model Context Protocol

## Channel Integrations

Telegram, Discord, Slack, WhatsApp, Email, Feishu, DingTalk, WeCom, QQ, Matrix, MoChat.

## Key Dependencies

| Library | Purpose |
|---------|---------|
| [LiteLLM](https://github.com/BerriAI/litellm) | Unified interface for 100+ LLM providers |
| [ChromaDB](https://www.trychroma.com) | Vector database for semantic memory search |
| [Whoosh](https://whoosh.readthedocs.io) | Full-text keyword search |
| [Ollama](https://ollama.ai) | Local embedding model inference |
| [Pydantic](https://docs.pydantic.dev) | Configuration validation |
| [loguru](https://github.com/Delgan/loguru) | Structured logging |
| [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io) | CLI input with history |
| [rich](https://rich.readthedocs.io) | Terminal formatting |

## Project Structure

```
nanobot/
├── agent/
│   ├── loop.py              # Orchestrator agent loop
│   ├── subagent.py          # Expert subagent manager
│   ├── context.py           # Prompt context builder
│   ├── expert_library.py    # Expert profile/memory management
│   ├── memory.py            # Base memory consolidation
│   ├── enhanced_memory/     # Vector search + semantic memory
│   │   ├── embeddings.py    # Ollama/OpenAI/local embedding providers
│   │   ├── vector_store.py  # ChromaDB storage
│   │   ├── search.py        # Hybrid search engine
│   │   └── chunkers.py      # Text chunking strategies
│   └── tools/               # Agent tools (fs, shell, web, MCP)
├── providers/               # LLM providers (LiteLLM, Azure, Codex)
├── channels/                # Chat platform integrations
├── bus/                     # Async message bus
├── config/                  # Pydantic config schema
├── session/                 # Session persistence
├── skills/                  # YAML+Markdown skill definitions
├── cron/                    # Task scheduling
└── cli/                     # CLI interface
```
