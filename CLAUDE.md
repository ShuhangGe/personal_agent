# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL — Workflow Rules (READ FIRST)

- **Start of session:** `git fetch origin && git merge origin/main`
- **Auto-commit after every code change** with Conventional Commits: `type(scope): description` (feat, fix, refactor, perf, docs, chore). No push unless asked.
- **Worktrees** live under `.claude/worktrees/<taskname>`, branch `<type>/<taskname>`. One per task, clean up when done.
- **Merge** (never rebase) worktree branches back to main. Resolve conflicts manually.
- Never commit secrets or `.env`.

### Solved Issues Log
Before debugging, check `TECHNICAL.md` for known solutions. After solving a non-obvious bug, add an entry: **Status / Symptom / Cause / Fix / Files**.

## Project Overview

Nanobot is a personal AI agent framework forked from https://github.com/HKUDS/nanobot. It uses an **orchestrator → subagent** architecture where a thin router delegates all real work to subagents.

## Core Architecture

### Three Roles

- **Orchestrator** — Routes tasks. Does NOT solve them. Only sees agent names + capability tags.
- **Expert** — Executes tasks. Has tools (filesystem, exec, web). Works in isolated workspace.
- **Evaluator** — Judges quality, learns user preferences, evolves over time. **The most important part.**

**Terminology:** "Subagent" = expert + evaluator (the whole unit). "Expert" = worker half only.

### Work Pipeline

```
1. User sends task
2. Orchestrator searches agent library
   ├── Found    → reuse existing subagent
   └── Not found → create new (name + description), tell user
3. Expert-evaluator round loop:
   │  Expert works → Evaluator reviews → Notify user per round
   │  User can interrupt anytime (silence = auto-continue)
   │  Exit when: evaluator satisfied OR max rounds reached
4. Result to user → User feedback:
   ├── Satisfied → done
   └── Not satisfied → Evaluator records bias in PREFERENCES.md
       → Re-enter round loop, reusing previous results
       → After revision limit (default 2-3), offer different approach
```

### Subagent Structure

```
agents/{name}/
├── AGENT.md              # Profile (description, capability tags)
├── WORKLOG.md            # Live work log
├── results/              # Task result files
├── expert/
│   ├── workspace/        # Isolated file sandbox
│   ├── sessions/         # Conversation history
│   └── memory/
│       ├── MEMORY.md     # What works
│       ├── HISTORY.md    # Task log
│       ├── SOUL.md       # Identity
│       └── EXPERIENCE.md # Lessons learned
└── evaluator/
    ├── workspace/        # Evaluator file sandbox
    ├── sessions/         # Evaluator history
    └── memory/
        ├── SOUL.md          # Evaluator identity
        ├── GUARDRAILS.md    # Quality rules (expert reads, evaluator writes)
        ├── PREFERENCES.md  # User preferences (learned from feedback)
        └── EXPERIENCE.md   # Evaluation history
```

**Naming:** Short, descriptive, kebab-case (e.g. `web-scraper`). No random hashes.

### Evaluator: Two Functions

1. **Evaluate expert work** — structured feedback, maintains GUARDRAILS.md with domain quality criteria
2. **Learn user preferences** — when user rejects a result the evaluator approved, it reflects on what it missed and records it in PREFERENCES.md

### Key Files

- `nanobot/agent/loop.py` — Orchestrator (`AgentLoop`)
- `nanobot/agent/subagent.py` — Subagent manager (`SubagentManager`)
- `nanobot/agent/agent_library.py` — Agent profiles and storage (`AgentLibrary`)
- `nanobot/agent/context.py` — System prompt builder (`ContextBuilder`)
- `nanobot/agent/memory.py` — Base memory consolidation
- `nanobot/agent/enhanced_memory/` — Vector search (Ollama embeddings + sqlite-vec)
- `nanobot/agent/tools/base.py` — Tool system (`Tool` + `ToolRegistry`)
- `nanobot/config/schema.py` — Pydantic config models

### Supporting Systems

- **Message Bus** (`nanobot/bus/`): Async queues. InboundMessage → agent → OutboundMessage.
- **Channels** (`nanobot/channels/`): Telegram, Discord, Slack, WhatsApp, Email, Feishu, DingTalk, WeCom, QQ, Matrix, MoChat.
- **Providers** (`nanobot/providers/`): `LiteLLMProvider` (100+ providers via litellm). Auto-detection by model name.
- **Skills** (`nanobot/skills/`): YAML-frontmatter + Markdown instruction files.
- **Sessions** (`nanobot/session/`): JSON conversation history. Expert and evaluator have separate sessions.

## Running and Testing

```bash
python3 -m nanobot                    # Run (CLI mode)
python3 -m pytest tests/ -v           # Run tests
python3 -m pytest tests/test_X.py -v  # Single test
```

Config: `~/.nanobot/config.json`. Workspace: `~/.nanobot/workspace`.

## Conventions

- **Async-first**: All I/O is async. `asyncio.create_task` for background work.
- **Workspace isolation**: Expert and evaluator each have separate sandboxed workspaces.
- **Type hints throughout**: Pydantic for config, dataclasses for events.
- **Loguru**: `from loguru import logger` everywhere.
- **No duplicates**: Modify existing files in-place. Never create parallel implementations.
