"""Agent library for managing subagent profiles, memory, worklogs, and results.

Each subagent has an expert (executes tasks) and an evaluator (reviews quality).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from nanobot.utils.helpers import ensure_dir


class AgentLibrary:
    """
    Manages the subagent library on disk.

    Directory layout per subagent:
        agents/{name}/
            AGENT.md            # Profile: description, approach, tags
            WORKLOG.md          # Live work log (plan + progress, updated during execution)
            results/
                {timestamp}.md  # Detailed result files
            expert/
                workspace/      # Expert's isolated file sandbox
                sessions/       # Persistent conversation history
                memory/
                    MEMORY.md, SOUL.md, EXPERIENCE.md, HISTORY.md
            evaluator/
                workspace/      # Evaluator's isolated file sandbox
                sessions/       # Evaluator conversation history
                memory/
                    MEMORY.md, SOUL.md, EXPERIENCE.md, GUARDRAILS.md
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.agents_dir = workspace / "agents"
        self.registry_file = self.agents_dir / "_registry.json"
        self._migrate_experts_to_agents()

    # ── Legacy migration ──────────────────────────────────────────────────

    def _migrate_experts_to_agents(self) -> None:
        """One-time migration: rename experts/ directory to agents/."""
        old_dir = self.workspace / "experts"
        if old_dir.exists() and not self.agents_dir.exists():
            import shutil
            shutil.move(str(old_dir), str(self.agents_dir))
            logger.info("Migrated {}/experts/ → {}/agents/", self.workspace, self.workspace)

    # ── Registry ──────────────────────────────────────────────────────────

    def load_registry(self) -> dict[str, dict]:
        if self.registry_file.exists():
            try:
                return json.loads(self.registry_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt agent registry, rebuilding")
        return {}

    def save_registry(self, registry: dict) -> None:
        ensure_dir(self.agents_dir)
        self.registry_file.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def update_registry_entry(self, name: str, **updates: object) -> None:
        registry = self.load_registry()
        entry = registry.get(name, {})
        entry.update(updates)
        registry[name] = entry
        self.save_registry(registry)

    # ── Agent directory helpers ───────────────────────────────────────────

    def get_agent_dir(self, name: str) -> Path:
        return self.agents_dir / name

    def agent_exists(self, name: str) -> bool:
        return (self.get_agent_dir(name) / "AGENT.md").exists()

    def list_agents(self) -> list[dict]:
        """List all subagents with their registry metadata."""
        registry = self.load_registry()
        agents = []
        if not self.agents_dir.exists():
            return agents
        for d in sorted(self.agents_dir.iterdir()):
            if d.is_dir() and (d / "AGENT.md").exists():
                meta = registry.get(d.name, {})
                agents.append({"name": d.name, **meta})
        return agents

    # ── Expert-half workspace helpers ─────────────────────────────────────

    def get_expert_workspace(self, name: str) -> Path:
        """Return the expert's isolated workspace directory, creating it if needed."""
        return ensure_dir(self.get_agent_dir(name) / "expert" / "workspace")

    def get_expert_sessions_dir(self, name: str) -> Path:
        """Return the expert's sessions directory, creating it if needed."""
        return ensure_dir(self.get_agent_dir(name) / "expert" / "sessions")

    # ── Agent profile CRUD ────────────────────────────────────────────────

    def load_agent_profile(self, name: str) -> str | None:
        path = self.get_agent_dir(name) / "AGENT.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def save_agent_profile(self, name: str, content: str) -> None:
        agent_dir = ensure_dir(self.get_agent_dir(name))
        (agent_dir / "AGENT.md").write_text(content, encoding="utf-8")

    # ── Expert-half memory ────────────────────────────────────────────────

    def _memory_dir(self, name: str) -> Path:
        return ensure_dir(self.get_agent_dir(name) / "expert" / "memory")

    def load_expert_memory(self, name: str) -> str:
        path = self._memory_dir(name) / "MEMORY.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def save_expert_memory(self, name: str, content: str) -> None:
        (self._memory_dir(name) / "MEMORY.md").write_text(content, encoding="utf-8")

    def append_expert_history(self, name: str, entry: str) -> None:
        path = self._memory_dir(name) / "HISTORY.md"
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    # ── Expert-half soul (identity/personality) ───────────────────────────

    def load_expert_soul(self, name: str) -> str:
        """Load the expert's soul (identity and personality)."""
        path = self._memory_dir(name) / "SOUL.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def save_expert_soul(self, name: str, content: str) -> None:
        """Save the expert's soul (identity and personality)."""
        (self._memory_dir(name) / "SOUL.md").write_text(content, encoding="utf-8")

    def init_expert_soul(self, name: str) -> None:
        """Initialize a default soul for a new expert."""
        soul_content = f"""# {name} Identity

## Personality
- I am methodical and detail-oriented
- I prefer structured outputs
- I focus on actionable insights, not just observations

## Expertise
- Specialized in tasks matching my tags
- Continuously learning from each task execution

## Constraints
- Stay focused on the assigned task
- Provide clear, actionable results
- Communicate progress through the work log

## Communication Style
- Concise and clear
- Results-oriented
- Transparent about challenges and solutions
"""
        self.save_expert_soul(name, soul_content)

    # ── Expert-half experience (lessons learned) ──────────────────────────

    def load_expert_experience(self, name: str) -> str:
        """Load the expert's experience (lessons learned from previous runs)."""
        path = self._memory_dir(name) / "EXPERIENCE.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def save_expert_experience(self, name: str, content: str) -> None:
        """Save the expert's experience (lessons learned)."""
        (self._memory_dir(name) / "EXPERIENCE.md").write_text(content, encoding="utf-8")

    def append_expert_experience(self, name: str, entry: str) -> None:
        """Append a new experience entry to the expert's experience file."""
        path = self._memory_dir(name) / "EXPERIENCE.md"
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def init_expert_experience(self, name: str) -> None:
        """Initialize an empty experience file for a new expert."""
        experience_content = """# Learned Experience

This file contains lessons learned from previous task executions.

---
"""
        self.save_expert_experience(name, experience_content)

    # ── Evaluator-half directory helpers ──────────────────────────────────

    def get_evaluator_workspace(self, name: str) -> Path:
        """Return the evaluator's isolated workspace directory, creating it if needed."""
        return ensure_dir(self.get_agent_dir(name) / "evaluator" / "workspace")

    def get_evaluator_sessions_dir(self, name: str) -> Path:
        """Return the evaluator's sessions directory, creating it if needed."""
        return ensure_dir(self.get_agent_dir(name) / "evaluator" / "sessions")

    def _evaluator_memory_dir(self, name: str) -> Path:
        return ensure_dir(self.get_agent_dir(name) / "evaluator" / "memory")

    # ── Evaluator-half memory ─────────────────────────────────────────────

    def load_evaluator_memory(self, name: str) -> str:
        path = self._evaluator_memory_dir(name) / "MEMORY.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def save_evaluator_memory(self, name: str, content: str) -> None:
        (self._evaluator_memory_dir(name) / "MEMORY.md").write_text(content, encoding="utf-8")

    def load_evaluator_soul(self, name: str) -> str:
        path = self._evaluator_memory_dir(name) / "SOUL.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def save_evaluator_soul(self, name: str, content: str) -> None:
        (self._evaluator_memory_dir(name) / "SOUL.md").write_text(content, encoding="utf-8")

    def init_evaluator_soul(self, name: str) -> None:
        """Initialize a default soul for the evaluator — critical reviewer persona."""
        soul_content = f"""# {name} Evaluator Identity

## Personality
- I am a thorough and critical reviewer
- I focus on correctness, completeness, and edge cases
- I do not accept mediocre work — I push for excellence
- I am specific about what needs improvement

## Review Criteria
- **Correctness**: Does the output accomplish the stated task without errors?
- **Completeness**: Are all aspects of the task addressed? Nothing missing?
- **Edge Cases**: Were potential edge cases considered and handled?
- **Clarity**: Is the output clear and well-structured?
- **Quality**: Is the work of high professional quality?

## Communication Style
- Direct and constructive
- Specific about issues found
- Acknowledge good work when present
- Always end with a clear verdict

## Verdict Rules
After reviewing, I MUST include a verdict block in my response:

---VERDICT---
Status: GOOD
---END VERDICT---

OR:

---VERDICT---
Status: NOT GOOD
Issues: [comma-separated list of issues]
---END VERDICT---

I use "GOOD" only when the output meets all criteria.
I use "NOT GOOD" with specific feedback when improvements are needed.
"""
        self.save_evaluator_soul(name, soul_content)

    def load_evaluator_experience(self, name: str) -> str:
        path = self._evaluator_memory_dir(name) / "EXPERIENCE.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def save_evaluator_experience(self, name: str, content: str) -> None:
        (self._evaluator_memory_dir(name) / "EXPERIENCE.md").write_text(content, encoding="utf-8")

    def append_evaluator_experience(self, name: str, entry: str) -> None:
        path = self._evaluator_memory_dir(name) / "EXPERIENCE.md"
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def init_evaluator_experience(self, name: str) -> None:
        experience_content = """# Learned Experience (Evaluator)

This file contains lessons learned from evaluating expert outputs.

---
"""
        self.save_evaluator_experience(name, experience_content)

    # ── Evaluator guardrails (read-only for expert) ──────────────────────

    def load_evaluator_guardrails(self, name: str) -> str:
        """Load the evaluator-maintained guardrails for this subagent."""
        path = self._evaluator_memory_dir(name) / "GUARDRAILS.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def save_evaluator_guardrails(self, name: str, content: str) -> None:
        """Save updated guardrails (only the evaluator should call this)."""
        (self._evaluator_memory_dir(name) / "GUARDRAILS.md").write_text(content, encoding="utf-8")

    def init_evaluator_guardrails(self, name: str) -> None:
        """Initialize an empty guardrails file for a new subagent."""
        content = f"""# Guardrails for {name}

This file is maintained by the evaluator. The expert MUST follow these rules
but CANNOT modify this file. Updated after every evaluation.

## Failed Approaches (Do NOT Repeat)


## Anti-Patterns


## Quality Standards

"""
        self.save_evaluator_guardrails(name, content)

    # ── Migration ────────────────────────────────────────────────────────

    def _migrate_flat_to_nested(self, name: str) -> None:
        """Migrate old flat layout to nested expert/evaluator layout.

        Old: agents/{name}/workspace/, agents/{name}/memory/, agents/{name}/sessions/
        New: agents/{name}/expert/workspace/, agents/{name}/expert/memory/, etc.
            + agents/{name}/evaluator/workspace/, agents/{name}/evaluator/memory/, etc.
        """
        agent_dir = self.get_agent_dir(name)
        if not agent_dir.exists():
            return

        old_workspace = agent_dir / "workspace"
        old_memory = agent_dir / "memory"
        old_sessions = agent_dir / "sessions"

        new_expert_workspace = agent_dir / "expert" / "workspace"
        new_expert_memory = agent_dir / "expert" / "memory"
        new_expert_sessions = agent_dir / "expert" / "sessions"

        if new_expert_workspace.exists():
            return

        import shutil

        if old_workspace.exists():
            shutil.move(str(old_workspace), str(new_expert_workspace))
        if old_memory.exists():
            shutil.move(str(old_memory), str(new_expert_memory))
        if old_sessions.exists():
            shutil.move(str(old_sessions), str(new_expert_sessions))

        self._init_evaluator_dirs(name)
        logger.info("Migrated agent '{}' from flat to nested layout", name)

    def _init_evaluator_dirs(self, name: str) -> None:
        """Create and initialize evaluator subdirectories and memory files.

        Only writes defaults when they don't already exist, so migrated data
        from temp subagents is preserved.
        """
        ensure_dir(self.get_evaluator_workspace(name))
        ensure_dir(self.get_evaluator_sessions_dir(name))
        mem_dir = self._evaluator_memory_dir(name)
        if not (mem_dir / "SOUL.md").exists():
            self.init_evaluator_soul(name)
        if not (mem_dir / "EXPERIENCE.md").exists():
            self.init_evaluator_experience(name)
        if not (mem_dir / "GUARDRAILS.md").exists():
            self.init_evaluator_guardrails(name)

    # ── Worklog ──────────────────────────────────────────────────────────

    def write_worklog(self, name: str, content: str) -> Path:
        path = self.get_agent_dir(name) / "WORKLOG.md"
        ensure_dir(path.parent)
        path.write_text(content, encoding="utf-8")
        return path

    def read_worklog(self, name: str) -> str | None:
        path = self.get_agent_dir(name) / "WORKLOG.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def get_worklog_path(self, name: str) -> Path:
        return self.get_agent_dir(name) / "WORKLOG.md"

    # ── Results ──────────────────────────────────────────────────────────

    def save_result(self, name: str, content: str) -> Path:
        """Save a timestamped result file. Returns the file path."""
        results_dir = ensure_dir(self.get_agent_dir(name) / "results")
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        path = results_dir / f"{ts}.md"
        path.write_text(content, encoding="utf-8")
        return path

    def list_results(self, name: str) -> list[Path]:
        results_dir = self.get_agent_dir(name) / "results"
        if not results_dir.exists():
            return []
        return sorted(results_dir.glob("*.md"), reverse=True)

    # ── Orchestrator summary ─────────────────────────────────────────────

    def build_agent_summary(self) -> str:
        """Build an XML summary of all subagents for the orchestrator's system prompt."""
        agents = self.list_agents()
        if not agents:
            return ""

        def esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<agent_library>"]
        for a in agents:
            name = esc(a["name"])
            desc = esc(a.get("description", a["name"]))
            times = a.get("times_used", 0)
            last = a.get("last_used", "never")
            tags = esc(a.get("tags", ""))
            lines.append(f'  <agent>')
            lines.append(f'    <name>{name}</name>')
            lines.append(f'    <description>{desc}</description>')
            lines.append(f'    <times_used>{times}</times_used>')
            lines.append(f'    <last_used>{last}</last_used>')
            lines.append(f'    <tags>{tags}</tags>')
            lines.append(f'  </agent>')
        lines.append("</agent_library>")
        return "\n".join(lines)

    # ── Agent creation helpers ────────────────────────────────────────────

    def create_agent(
        self,
        name: str,
        description: str,
        tags: list[str],
        task: str,
        approach: str,
        tools_used: list[str] | None = None,
        skills_used: list[str] | None = None,
    ) -> None:
        """Create a new subagent with profile, expert memory/soul, evaluator dirs, and registry."""
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M")

        profile = f"""# {name}

## Description
{description}

## Approach
{approach}

## Created
{now_str}

## Past Tasks
- {task} ({now_str}, success)
"""
        self.save_agent_profile(name, profile)

        ensure_dir(self._memory_dir(name))
        mem_path = self._memory_dir(name) / "MEMORY.md"
        if not mem_path.exists():
            mem_path.write_text("", encoding="utf-8")
        hist_path = self._memory_dir(name) / "HISTORY.md"
        if not hist_path.exists():
            hist_path.write_text("", encoding="utf-8")

        self.init_expert_soul(name)
        self.init_expert_experience(name)

        ensure_dir(self.get_agent_dir(name) / "results")
        ensure_dir(self.get_agent_dir(name) / "expert" / "workspace")
        ensure_dir(self.get_agent_dir(name) / "expert" / "sessions")

        self._init_evaluator_dirs(name)

        self.update_registry_entry(
            name,
            description=description,
            tags=", ".join(tags),
            times_used=1,
            created=now.isoformat(),
            last_used=now.isoformat(),
            tools_used=tools_used or [],
            skills_used=skills_used or [],
        )

        logger.info("Created subagent: {}", name)

    def record_usage(self, name: str) -> None:
        """Increment times_used and update last_used for an existing subagent."""
        registry = self.load_registry()
        entry = registry.get(name, {})
        entry["times_used"] = entry.get("times_used", 0) + 1
        entry["last_used"] = datetime.now().isoformat()
        registry[name] = entry
        self.save_registry(registry)
