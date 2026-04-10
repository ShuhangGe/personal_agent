# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nanobot is a personal AI agent framework forked from https://github.com/HKUDS/nanobot, being customized into a personal agent. It uses an **orchestrator + subagent** architecture where a main agent loop delegates tasks to specialized subagents.

## Critical Architecture Concept

**A subagent is NOT an expert. A subagent HAS an expert and an evaluator.**

Each subagent is a paired unit with two roles:
- **Expert**: executes the actual task (has tools, writes files, runs code)
- **Evaluator**: reviews the expert's output, maintains guardrails, pushes for quality

The on-disk directory `agents/{name}/` represents a **subagent**, not an expert.

**Naming must be meaningful**: Every subagent must get a short, descriptive, English kebab-case name that relates to its task (e.g. `novel-analyzer`, `web-scraper`). Random hashes or task dumps as names are never acceptable.

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

### Orchestrator + Subagent Pattern

The system has two levels:

- **Orchestrator** (`nanobot/agent/loop.py`): The `AgentLoop` class. Receives messages from channels, builds context (history + memory + agent library), calls the LLM with a minimal tool set (`spawn`, `message`, `cron`), and delegates real work to subagents via `spawn`.

- **Subagents** (`nanobot/agent/subagent.py`): The `SubagentManager` class. Each subagent gets an isolated workspace, its own session history, and persistent memory. Subagents are spawned as background `asyncio.Task`s and announce results back via the message bus.

### Subagent Structure

Each subagent lives in `agents/{name}/` with this structure:
```
agents/{name}/
├── AGENT.md            # Profile (description, approach, tags)
├── WORKLOG.md          # Live work log during execution
├── results/
│   └── {timestamp}.md  # Detailed result files
├── expert/             # The expert half (executes tasks)
│   ├── workspace/      # Isolated file sandbox
│   ├── sessions/       # Persistent conversation history
│   └── memory/
│       ├── MEMORY.md   # Persistent knowledge (what works)
│       ├── HISTORY.md  # Grep-searchable task log
│       ├── SOUL.md     # Expert's identity and personality
│       └── EXPERIENCE.md # Lessons learned
└── evaluator/          # The evaluator half (reviews quality)
    ├── workspace/      # Evaluator's own file sandbox
    ├── sessions/       # Evaluator conversation history
    └── memory/
        ├── MEMORY.md   # Evaluation patterns
        ├── SOUL.md     # Evaluator personality
        ├── EXPERIENCE.md # Evaluation history
        └── GUARDRAILS.md # Rules the expert must follow (read-only for expert)
```

Managed by `AgentLibrary` in `nanobot/agent/agent_library.py`. Subagents are created automatically after first task completion via LLM-generated profiles.

### Memory System

Two layers coexist:

- **Base memory** (`nanobot/agent/memory.py`): `MemoryStore` (MEMORY.md + HISTORY.md file I/O) and `MemoryConsolidator` (LLM-based consolidation that triggers when context approaches token limits).

- **Enhanced memory** (`nanobot/agent/enhanced_memory/`): `EnhancedMemoryConsolidator` extends the base consolidator with vector search. Uses Ollama embeddings (`qwen3-embedding:0.6b`) for semantic similarity and sqlite-vec for storage. The agent loop initializes this by default, falling back to the base consolidator if Ollama is unavailable.

### Guardrails System

The evaluator maintains a `GUARDRAILS.md` file per subagent that captures:
- **Failed approaches** that the expert must not repeat
- **Anti-patterns** discovered during reviews
- **Quality standards** that must be maintained

The expert loads guardrails at the start of every run but cannot modify them. Only the evaluator updates guardrails after each review (both GOOD and NOT GOOD verdicts).

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

`nanobot/agent/context.py` — `ContextBuilder` assembles the system prompt from: identity files (AGENTS.md, SOUL.md, USER.md, TOOLS.md), long-term memory, agent library summary, and active skills.

### Skills

`nanobot/skills/` — Skills are YAML-frontmatter + Markdown instruction files (SKILL.md). Built-in skills include GitHub, weather, URL summarization, and tmux control. Loaded by `SkillsLoader`.

### Session Management

`nanobot/session/` — `SessionManager` persists conversation history as JSON. Each subagent's expert and evaluator have their own sessions that persist across runs.

## Configuration

`nanobot/config/schema.py` defines all config using Pydantic models. Key sections:
- `agents.defaults`: model, provider, context window, temperature
- `providers.*`: API keys and base URLs per provider
- `channels.*`: channel-specific configs
- `tools.*`: web search, exec, MCP servers

Config auto-migrates old formats. Provider matching uses keyword detection against model names.

## Key Conventions

- **Async-first**: All I/O is async. Use `asyncio.create_task` for background work.
- **Workspace isolation**: The expert half operates in `agents/{name}/expert/workspace/`. Filesystem tools are sandboxed to this directory. The evaluator has its own workspace under `agents/{name}/evaluator/workspace/`.
- **Type hints throughout**: Pydantic models for config, dataclasses for events, type annotations on all functions.
- **Loguru for logging**: `from loguru import logger` everywhere.
- **LLM tool calls**: Tool definitions use OpenAI function-calling format. The `chat_with_retry` method on providers handles retries.
- **Subagent lifecycle**: First spawn creates a temp subagent (`_task-{id}` dir). After completion, LLM generates a profile and the temp dir is migrated to a named subagent directory.
- **Terminology**: "subagent" or "agent" = the whole entity (expert + evaluator). "expert" = the worker half only. Never use "expert" to mean the whole subagent.
