"""Expert library for managing emergent expert profiles, memory, worklogs, and results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from nanobot.utils.helpers import ensure_dir


class ExpertLibrary:
    """
    Manages the expert library on disk.

    Directory layout per expert:
        experts/{name}/
            EXPERT.md           # Profile: description, approach, tags
            WORKLOG.md          # Live work log (plan + progress, updated during execution)
            memory/
                MEMORY.md       # Persistent knowledge across runs
                HISTORY.md      # Grep-searchable task log
            results/
                {timestamp}.md  # Detailed result files
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.experts_dir = workspace / "experts"
        self.registry_file = self.experts_dir / "_registry.json"

    # ── Registry ──────────────────────────────────────────────────────────

    def load_registry(self) -> dict[str, dict]:
        if self.registry_file.exists():
            try:
                return json.loads(self.registry_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt expert registry, rebuilding")
        return {}

    def save_registry(self, registry: dict) -> None:
        ensure_dir(self.experts_dir)
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
        return self.experts_dir / name

    def expert_exists(self, name: str) -> bool:
        return (self.get_expert_dir(name) / "EXPERT.md").exists()

    def list_experts(self) -> list[dict]:
        """List all experts with their registry metadata."""
        registry = self.load_registry()
        experts = []
        if not self.experts_dir.exists():
            return experts
        for d in sorted(self.experts_dir.iterdir()):
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
        return ensure_dir(self.get_expert_dir(name) / "memory")

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
        """Create a new expert with profile, empty memory, and registry entry."""
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

        ensure_dir(self._memory_dir(name))
        mem_path = self._memory_dir(name) / "MEMORY.md"
        if not mem_path.exists():
            mem_path.write_text("", encoding="utf-8")
        hist_path = self._memory_dir(name) / "HISTORY.md"
        if not hist_path.exists():
            hist_path.write_text("", encoding="utf-8")

        ensure_dir(self.get_expert_dir(name) / "results")

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
