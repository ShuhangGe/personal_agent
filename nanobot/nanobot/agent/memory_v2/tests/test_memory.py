"""Tests for the memory system."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from nanobot.agent.memory_v2.config import MemoryConfig
from nanobot.agent.memory_v2.manager import MemoryManager
from nanobot.agent.memory_v2.models import MemoryTier


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def memory_manager(temp_workspace):
    """Create a memory manager for testing."""
    config = MemoryConfig()  # Use default config
    manager = MemoryManager(workspace=temp_workspace, config=config)
    await manager.initialize()

    yield manager

    await manager.shutdown()


class TestWorkingMemory:
    """Tests for working memory store."""

    def test_add_message(self, memory_manager):
        """Test adding messages to working memory."""
        memory_id = memory_manager.add_to_working_memory(
            content="Test message", session_key="test_session"
        )

        assert memory_id is not None

        # Verify message was added
        working = memory_manager.get_working_memory("test_session")
        assert len(working.messages) == 1
        assert working.messages[0].content == "Test message"

    def test_search_working_memory(self, memory_manager):
        """Test searching working memory."""
        memory_manager.add_to_working_memory(
            content="Python is great", session_key="test_session"
        )
        memory_manager.add_to_working_memory(
            content="I love coding", session_key="test_session"
        )

        results = memory_manager.working.search("Python")
        assert len(results) == 1
        assert "Python" in results[0].content


class TestSessionMemory:
    """Tests for session memory store."""

    @pytest.mark.asyncio
    async def test_add_session_memory(self, memory_manager):
        """Test adding memories to session store."""
        memory_id = await memory_manager.remember(
            content="Important fact about the project",
            tier=MemoryTier.SESSION,
            metadata={"session_key": "test_session"},
            importance=0.8,
        )

        assert memory_id is not None

        # Search for the memory
        results = await memory_manager.search("project", tiers=[MemoryTier.SESSION])
        assert len(results) >= 1
        assert any("project" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_search_session_memory(self, memory_manager):
        """Test searching session memories."""
        # Add some memories
        await memory_manager.remember(
            content="User prefers dark mode", tier=MemoryTier.SESSION, importance=0.7
        )
        await memory_manager.remember(
            content="Project uses Python", tier=MemoryTier.SESSION, importance=0.6
        )
        await memory_manager.remember(
            content="Meeting tomorrow at 10am", tier=MemoryTier.SESSION, importance=0.5
        )

        # Search for "Python"
        results = await memory_manager.search("Python", tiers=[MemoryTier.SESSION])
        assert len(results) >= 1
        assert any("python" in r.content.lower() for r in results)


class TestLongTermMemory:
    """Tests for long-term memory store."""

    @pytest.mark.asyncio
    async def test_add_longterm_memory(self, memory_manager):
        """Test adding facts to long-term memory."""
        memory_id = await memory_manager.remember(
            content="User works as a software engineer",
            tier=MemoryTier.LONGTERM,
            metadata={"category": "profile"},
            importance=1.0,
        )

        assert memory_id is not None

        # Check MEMORY.md file
        memory_file = memory_manager.longterm.memory_file
        assert memory_file.exists()

        content = memory_file.read_text(encoding="utf-8")
        assert "software engineer" in content.lower()

    @pytest.mark.asyncio
    async def test_search_longterm_memory(self, memory_manager):
        """Test searching long-term memories."""
        # Add some facts
        await memory_manager.remember(
            content="User lives in San Francisco", tier=MemoryTier.LONGTERM, importance=1.0
        )
        await memory_manager.remember(
            content="User prefers async communication", tier=MemoryTier.LONGTERM, importance=0.9
        )

        # Search for "San Francisco"
        results = await memory_manager.search_facts("San Francisco")
        assert len(results) >= 1


class TestMemoryManager:
    """Tests for memory manager integration."""

    @pytest.mark.asyncio
    async def test_cross_tier_search(self, memory_manager):
        """Test searching across all tiers."""
        # Add to working memory
        memory_manager.add_to_working_memory(
            content="Python programming", session_key="test"
        )

        # Add to session memory
        await memory_manager.remember(
            content="Python is a great language", tier=MemoryTier.SESSION
        )

        # Add to long-term memory
        await memory_manager.remember(
            content="User knows Python well", tier=MemoryTier.LONGTERM
        )

        # Search across all tiers
        results = await memory_manager.search("Python")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_memory_stats(self, memory_manager):
        """Test getting memory statistics."""
        # Add some memories
        await memory_manager.remember(
            content="Test fact", tier=MemoryTier.SESSION, importance=0.5
        )

        stats = await memory_manager.get_stats()

        assert isinstance(stats, dict)
        assert MemoryTier.WORKING in stats
        assert MemoryTier.SESSION in stats
        assert MemoryTier.LONGTERM in stats


class TestMemoryConfig:
    """Tests for memory configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = MemoryConfig()

        assert config.tiers.working.max_messages == 100
        assert config.tiers.session.retention_days == 30
        assert config.embeddings.provider == "local"

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "tiers": {
                "working": {"max_messages": 50},
                "session": {"retention_days": 60},
            },
            "embeddings": {"provider": "openai", "model": "text-embedding-3-small"},
        }

        config = MemoryConfig.from_dict(config_dict)

        assert config.tiers.working.max_messages == 50
        assert config.tiers.session.retention_days == 60
        assert config.embeddings.provider == "openai"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
