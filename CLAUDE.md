# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nanobot is a personal AI agent framework forked from https://github.com/HKUDS/nanobot, being customized into a personal agent. It uses an **orchestrator + expert subagent** architecture where a main agent loop delegates tasks to specialized expert subagents that execute with isolated workspaces and persistent memory.

## Running and Testing

```bash
# Run the agent (CLI mode)
python3 -m nanobot

# Run tests
python3 -m pytest tests/ -v

# Run a single test file
python3 -m pytest tests/test_context_prompt_cache.py -v
```

Config lives at `~/.nanobot/config.json`. Workspace defaults to `~/.nanobot/workspace`.

## Architecture

### Orchestrator + Expert Pattern

The system has two levels of agents:

- **Orchestrator** (`nanobot/agent/loop.py`): The `AgentLoop` class. Receives messages from channels, builds context (history + memory + expert library), calls the LLM with a minimal tool set (`spawn`, `message`, `cron`), and delegates real work to expert subagents via `spawn`.

- **Expert Subagents** (`nanobot/agent/subagent.py`): The `SubagentManager` class. Each expert gets an isolated workspace, its own session history, and persistent memory files. Experts are spawned as background `asyncio.Task`s and announce results back via the message bus.

### Expert Memory and Workspace

Each expert lives in `experts/{name}/` with this structure:
```
experts/{name}/
├── EXPERT.md           # Profile (description, approach, tags)
├── WORKLOG.md          # Live work log during execution
├── memory/
│   ├── MEMORY.md       # Persistent knowledge (what works/doesn't)
│   ├── HISTORY.md      # Grep-searchable task log
│   ├── SOUL.md         # Expert's identity and personality
│   └── EXPERIENCE.md   # Lessons learned from previous runs
├── workspace/          # Isolated file sandbox
├── sessions/           # Persistent conversation history
└── results/            # Timestamped result files
```

Managed by `ExpertLibrary` in `nanobot/agent/expert_library.py`. Experts are created automatically after first task completion via LLM-generated profiles.

### Memory System

Two layers coexist:

- **Base memory** (`nanobot/agent/memory.py`): `MemoryStore` (MEMORY.md + HISTORY.md file I/O) and `MemoryConsolidator` (LLM-based consolidation that triggers when context approaches token limits).

- **Enhanced memory** (`nanobot/agent/enhanced_memory/`): `EnhancedMemoryConsolidator` extends the base consolidator with vector search. Uses Ollama embeddings (`qwen3-embedding:0.6b`) for semantic similarity and sqlite-vec for storage. The agent loop initializes this by default, falling back to the base consolidator if Ollama is unavailable.

### Message Bus

`nanobot/bus/` — async queue-based decoupling between channels and the agent loop. `InboundMessage` flows from channels to agent, `OutboundMessage` flows back.

### Tool System

Tools are subclasses of `Tool` (`nanobot/agent/tools/base.py`) registered in a `ToolRegistry`. Each tool defines `name`, `description`, `parameters` (JSON Schema), and an async `execute()` method. The registry handles schema generation, parameter casting, and execution.

**Orchestrator tools**: `spawn`, `message`, `cron` (minimal set for delegation).
**Expert tools**: `read_file`, `write_file`, `edit_file`, `list_dir`, `exec`, `web_search`, `web_fetch` (full set for task execution).

### Channel System

`nanobot/channels/` — All channels inherit from `BaseChannel`. Supported: Telegram, Discord, Slack, WhatsApp, Email, Feishu, DingTalk, WeCom, QQ, Matrix, MoChat. Channel configs are stored as extra dict fields in `ChannelsConfig` (Pydantic `extra="allow"`).

### Provider System

`nanobot/providers/` — All LLM calls go through `LiteLLMProvider` (supports 100+ providers via litellm). Specialized providers exist for Azure OpenAI, OpenAI Codex, and custom endpoints. Provider auto-detection is based on model name keywords and configured API keys (see `Config._match_provider()` in `nanobot/config/schema.py`).

### Context Building

`nanobot/agent/context.py` — `ContextBuilder` assembles the system prompt from: identity files (AGENTS.md, SOUL.md, USER.md, TOOLS.md), long-term memory, expert library summary, and active skills.

### Skills

`nanobot/skills/` — Skills are YAML-frontmatter + Markdown instruction files (SKILL.md). Built-in skills include GitHub, weather, URL summarization, and tmux control. Loaded by `SkillsLoader`.

### Session Management

`nanobot/session/` — `SessionManager` persists conversation history as JSON. Each expert has its own session that persists across runs.

## Configuration

`nanobot/config/schema.py` defines all config using Pydantic models. Key sections:
- `agents.defaults`: model, provider, context window, temperature
- `providers.*`: API keys and base URLs per provider
- `channels.*`: channel-specific configs
- `tools.*`: web search, exec, MCP servers

Config auto-migrates old formats. Provider matching uses keyword detection against model names.

## Key Conventions

- **Async-first**: All I/O is async. Use `asyncio.create_task` for background work.
- **Workspace isolation**: Experts operate in `experts/{name}/workspace/`. Filesystem tools are sandboxed to this directory.
- **Type hints throughout**: Pydantic models for config, dataclasses for events, type annotations on all functions.
- **Loguru for logging**: `from loguru import logger` everywhere.
- **LLM tool calls**: Tool definitions use OpenAI function-calling format. The `chat_with_retry` method on providers handles retries.
- **Expert lifecycle**: First spawn creates a generic expert (`_task-{id}` temp dir). After completion, LLM generates a profile and the temp dir is migrated to a named expert directory.
