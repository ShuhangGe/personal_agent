"""Expert library for managing emergent expert profiles, memory, worklogs, and results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from nanobot.utils.helpers import ensure_dir


class ExpertLibrary:
    """
    Manages the agent library on disk.

    Directory layout per agent:
        agents/{name}/
            EXPERT.md           # Profile: description, approach, tags
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
                    MEMORY.md, SOUL.md, EXPERIENCE.md
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.agents_dir = workspace / "agents"
        self.registry_file = self.agents_dir / "_registry.json"
        # Auto-migrate old experts/ directory to agents/
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
                logger.warning("Corrupt expert registry, rebuilding")
        return {}

    def save_registry(self, registry: dict) -> None:
        ensure_dir(self.agents_dir)
        self.registry_file.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def update_registry_entry(self, expert_name: str, **updates: object) -> None:
        registry = self.load_registry()
        entry = registry.get(expert_name, {})
        entry.update(updates)
        registry[expert_name] = entry
        self.save_registry(registry)

    # ── Expert directory helpers ──────────────────────────────────────────

    def get_expert_dir(self, name: str) -> Path:
        return self.agents_dir / name

    def expert_exists(self, name: str) -> bool:
        return (self.get_expert_dir(name) / "EXPERT.md").exists()

    def get_expert_workspace(self, name: str) -> Path:
        """Return the expert's isolated workspace directory, creating it if needed."""
        return ensure_dir(self.get_expert_dir(name) / "expert" / "workspace")

    def get_expert_sessions_dir(self, name: str) -> Path:
        """Return the expert's sessions directory, creating it if needed."""
        return ensure_dir(self.get_expert_dir(name) / "expert" / "sessions")

    def list_experts(self) -> list[dict]:
        """List all experts with their registry metadata."""
        registry = self.load_registry()
        experts = []
        if not self.agents_dir.exists():
            return experts
        for d in sorted(self.agents_dir.iterdir()):
            if d.is_dir() and (d / "EXPERT.md").exists():
                meta = registry.get(d.name, {})
                experts.append({"name": d.name, **meta})
        return experts

    # ── Profile CRUD ─────────────────────────────────────────────────────

    def load_expert_profile(self, name: str) -> str | None:
        path = self.get_expert_dir(name) / "EXPERT.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def save_expert_profile(self, name: str, content: str) -> None:
        expert_dir = ensure_dir(self.get_expert_dir(name))
        (expert_dir / "EXPERT.md").write_text(content, encoding="utf-8")

    # ── Per-expert memory ────────────────────────────────────────────────

    def _memory_dir(self, name: str) -> Path:
        return ensure_dir(self.get_expert_dir(name) / "expert" / "memory")

    # ── Evaluator directory helpers ─────────────────────────────────────

    def get_evaluator_workspace(self, name: str) -> Path:
        """Return the evaluator's isolated workspace directory, creating it if needed."""
        return ensure_dir(self.get_expert_dir(name) / "evaluator" / "workspace")

    def get_evaluator_sessions_dir(self, name: str) -> Path:
        """Return the evaluator's sessions directory, creating it if needed."""
        return ensure_dir(self.get_expert_dir(name) / "evaluator" / "sessions")

    def _evaluator_memory_dir(self, name: str) -> Path:
        return ensure_dir(self.get_expert_dir(name) / "evaluator" / "memory")

    # ── Evaluator memory ────────────────────────────────────────────────

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

    # ── Migration ────────────────────────────────────────────────────────

    def _migrate_flat_to_nested(self, name: str) -> None:
        """Migrate old flat layout to nested expert/evaluator layout.

        Old: agents/{name}/workspace/, agents/{name}/memory/, agents/{name}/sessions/
        New: agents/{name}/expert/workspace/, agents/{name}/expert/memory/, etc.
            + agents/{name}/evaluator/workspace/, agents/{name}/evaluator/memory/, etc.
        """
        expert_dir = self.get_expert_dir(name)
        if not expert_dir.exists():
            return

        # Only migrate if old flat layout exists and new layout doesn't
        old_workspace = expert_dir / "workspace"
        old_memory = expert_dir / "memory"
        old_sessions = expert_dir / "sessions"

        new_expert_workspace = expert_dir / "expert" / "workspace"
        new_expert_memory = expert_dir / "expert" / "memory"
        new_expert_sessions = expert_dir / "expert" / "sessions"

        # Skip if already migrated
        if new_expert_workspace.exists():
            return

        import shutil

        # Move workspace → expert/workspace
        if old_workspace.exists():
            shutil.move(str(old_workspace), str(new_expert_workspace))

        # Move memory → expert/memory
        if old_memory.exists():
            shutil.move(str(old_memory), str(new_expert_memory))

        # Move sessions → expert/sessions
        if old_sessions.exists():
            shutil.move(str(old_sessions), str(new_expert_sessions))

        # Create evaluator subdirs with initialized files
        self._init_evaluator_dirs(name)

        logger.info("Migrated expert '{}' from flat to nested layout", name)

    def _init_evaluator_dirs(self, name: str) -> None:
        """Create and initialize evaluator subdirectories and memory files.

        Only writes default SOUL.md / EXPERIENCE.md when they don't already
        exist, so migrated data from temp experts is preserved.
        """
        ensure_dir(self.get_evaluator_workspace(name))
        ensure_dir(self.get_evaluator_sessions_dir(name))
        mem_dir = self._evaluator_memory_dir(name)
        if not (mem_dir / "SOUL.md").exists():
            self.init_evaluator_soul(name)
        if not (mem_dir / "EXPERIENCE.md").exists():
            self.init_evaluator_experience(name)

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

    # ── Per-expert soul (identity/personality) ─────────────────────────────

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

    # ── Per-expert experience (lessons learned) ────────────────────────────

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

    # ── Worklog ──────────────────────────────────────────────────────────

    def write_worklog(self, name: str, content: str) -> Path:
        path = self.get_expert_dir(name) / "WORKLOG.md"
        ensure_dir(path.parent)
        path.write_text(content, encoding="utf-8")
        return path

    def read_worklog(self, name: str) -> str | None:
        path = self.get_expert_dir(name) / "WORKLOG.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def get_worklog_path(self, name: str) -> Path:
        return self.get_expert_dir(name) / "WORKLOG.md"

    # ── Results ──────────────────────────────────────────────────────────

    def save_result(self, name: str, content: str) -> Path:
        """Save a timestamped result file. Returns the file path."""
        results_dir = ensure_dir(self.get_expert_dir(name) / "results")
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        path = results_dir / f"{ts}.md"
        path.write_text(content, encoding="utf-8")
        return path

    def list_results(self, name: str) -> list[Path]:
        results_dir = self.get_expert_dir(name) / "results"
        if not results_dir.exists():
            return []
        return sorted(results_dir.glob("*.md"), reverse=True)

    # ── Orchestrator summary ─────────────────────────────────────────────

    def build_expert_summary(self) -> str:
        """Build an XML summary of all experts for the orchestrator's system prompt."""
        experts = self.list_experts()
        if not experts:
            return ""

        def esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<expert_library>"]
        for e in experts:
            name = esc(e["name"])
            desc = esc(e.get("description", e["name"]))
            times = e.get("times_used", 0)
            last = e.get("last_used", "never")
            tags = esc(e.get("tags", ""))
            lines.append(f'  <expert>')
            lines.append(f'    <name>{name}</name>')
            lines.append(f'    <description>{desc}</description>')
            lines.append(f'    <times_used>{times}</times_used>')
            lines.append(f'    <last_used>{last}</last_used>')
            lines.append(f'    <tags>{tags}</tags>')
            lines.append(f'  </expert>')
        lines.append("</expert_library>")
        return "\n".join(lines)

    # ── Expert creation helpers ──────────────────────────────────────────

    def create_expert(
        self,
        name: str,
        description: str,
        tags: list[str],
        task: str,
        approach: str,
        tools_used: list[str] | None = None,
        skills_used: list[str] | None = None,
    ) -> None:
        """Create a new expert with profile, empty memory, soul, experience, and registry entry."""
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
        self.save_expert_profile(name, profile)

        # Initialize memory files
        ensure_dir(self._memory_dir(name))
        mem_path = self._memory_dir(name) / "MEMORY.md"
        if not mem_path.exists():
            mem_path.write_text("", encoding="utf-8")
        hist_path = self._memory_dir(name) / "HISTORY.md"
        if not hist_path.exists():
            hist_path.write_text("", encoding="utf-8")

        # Initialize soul and experience
        self.init_expert_soul(name)
        self.init_expert_experience(name)

        ensure_dir(self.get_expert_dir(name) / "results")
        ensure_dir(self.get_expert_dir(name) / "expert" / "workspace")
        ensure_dir(self.get_expert_dir(name) / "expert" / "sessions")

        # Initialize evaluator dirs
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

        logger.info("Created expert profile: {}", name)

    def record_usage(self, name: str) -> None:
        """Increment times_used and update last_used for an existing expert."""
        registry = self.load_registry()
        entry = registry.get(name, {})
        entry["times_used"] = entry.get("times_used", 0) + 1
        entry["last_used"] = datetime.now().isoformat()
        registry[name] = entry
        self.save_registry(registry)
