"""Memory consolidation engine - intelligent summarization and tier promotion."""

from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.memory_v2.embeddings import EmbeddingProvider
from nanobot.agent.memory_v2.entities import EntityExtractor
from nanobot.agent.memory_v2.longterm import LongTermMemoryStore
from nanobot.agent.memory_v2.models import (
    SessionMemory,
    WorkingMemory,
)


class MemoryConsolidator:
    """
    Intelligent memory consolidation engine.

    Consolidates working memory to session memory and potentially
    to long-term memory based on importance and content analysis.
    """

    def __init__(
        self,
        workspace: Path,
        embedder: EmbeddingProvider,
        entity_extractor: EntityExtractor,
    ) -> None:
        """
        Initialize consolidator.

        Args:
            workspace: Workspace directory
            embedder: Embedding provider
            entity_extractor: Entity extraction service
        """
        self.workspace = workspace
        self.embedder = embedder
        self.entity_extractor = entity_extractor
        self.longterm = LongTermMemoryStore(workspace)

    async def initialize(self) -> None:
        """Initialize consolidator components."""
        await self.longterm.initialize(embedder=self.embedder)

    async def consolidate_to_session(
        self,
        session_key: str,
        working_memory: WorkingMemory,
        importance_threshold: float = 0.6,
    ) -> SessionMemory:
        """
        Consolidate working memory to session memory.

        Args:
            session_key: Session identifier
            working_memory: Working memory to consolidate
            importance_threshold: Minimum importance for promotion to long-term

        Returns:
            Created session memory
        """
        if not working_memory.messages:
            return None

        # 1. Summarize conversation
        summary = await self._summarize_conversation(working_memory)

        # 2. Extract entities
        entities = await self._extract_entities(working_memory)

        # 3. Classify topics
        topics = self._classify_topics(summary, entities)

        # 4. Generate embedding
        embedding = await self.embedder.embed(summary) if self.embedder else None

        # 5. Calculate importance
        importance = await self._calculate_importance(summary, entities, topics)

        # 6. Create session memory
        session_memory = SessionMemory(
            session_key=session_key,
            summary=summary,
            embedding=embedding,
            entities=entities,
            topics=topics,
            importance_score=importance,
        )

        return session_memory

    async def _summarize_conversation(self, working_memory: WorkingMemory) -> str:
        """
        Summarize conversation from working memory.

        For now, this is a simple concatenation. In a full implementation,
        this would use an LLM to generate a summary.
        """
        messages = working_memory.messages[-10:]  # Last 10 messages
        summary_parts = []

        for msg in messages:
            role = msg.role.upper()
            content = msg.content[:200]  # Truncate long messages
            summary_parts.append(f"{role}: {content}")

        return "\n".join(summary_parts)

    async def _extract_entities(self, working_memory: WorkingMemory) -> list:
        """Extract entities from working memory."""
        if not self.entity_extractor.enabled:
            return []

        all_entities = []
        seen_entities = set()

        for msg in working_memory.messages:
            entities = self.entity_extractor.extract(msg.content)
            for entity in entities:
                # Avoid duplicates
                key = (entity.name, entity.type)
                if key not in seen_entities:
                    seen_entities.add(key)
                    all_entities.append(entity)

        return all_entities

    def _classify_topics(self, summary: str, entities: list) -> list[str]:
        """
        Classify topics from summary and entities.

        Simple rule-based classification.
        """
        topics = []
        summary_lower = summary.lower()

        # Topic keywords
        topic_keywords = {
            "user_preference": ["prefer", "like", "dislike", "want"],
            "decision": ["decided", "chose", "selected", "agreed"],
            "relationship": ["know", "friend", "colleague", "team"],
            "project": ["project", "feature", "bug", "implement"],
            "question": ["how", "what", "why", "when", "where"],
            "technical": ["code", "function", "api", "database", "server"],
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in summary_lower for keyword in keywords):
                topics.append(topic)

        # Add entity types as topics
        entity_types = {e.type for e in entities}
        topics.extend(entity_types)

        return topics[:5]  # Limit to 5 topics

    async def _calculate_importance(
        self,
        summary: str,
        entities: list,
        topics: list,
    ) -> float:
        """
        Calculate importance score for consolidation.

        Args:
            summary: Conversation summary
            entities: Extracted entities
            topics: Classified topics

        Returns:
            Importance score (0.0 to 1.0)
        """
        score = 0.0

        # Entity density (more entities = more important)
        if entities:
            score += min(len(entities) / 10, 0.3)

        # Topic importance
        important_topics = {"user_preference", "decision", "relationship"}
        if any(t in important_topics for t in topics):
            score += 0.3

        # Length and detail
        if len(summary) > 500:
            score += 0.2

        # Named entities
        if entities:
            score += 0.2

        return min(score, 1.0)

    async def consolidate_to_longterm(
        self,
        session_memory: SessionMemory,
        importance_threshold: float = 0.7,
    ) -> None:
        """
        Extract important facts from session to long-term memory.

        Args:
            session_memory: Session memory to extract facts from
            importance_threshold: Minimum importance to promote
        """
        if session_memory.importance_score < importance_threshold:
            return

        # Extract facts from summary
        facts = self._extract_facts(session_memory)

        # Store each fact in long-term memory
        for fact, importance in facts:
            await self.longterm.add(
                fact=fact,
                metadata={"category": self._categorize_fact(fact)},
                importance=importance,
            )

    def _extract_facts(self, session_memory: SessionMemory) -> list[tuple[str, float]]:
        """
        Extract important facts from session memory.

        Simple rule-based extraction. In a full implementation,
        this would use an LLM to extract structured facts.
        """
        facts = []
        summary = session_memory.summary

        # Split into sentences
        sentences = [s.strip() for s in summary.split(".") if s.strip()]

        for sentence in sentences:
            # Skip short sentences
            if len(sentence) < 20:
                continue

            # Check for fact indicators
            if any(
                indicator in sentence.lower()
                for indicator in ["is", "are", "has", "uses", "prefers", "wants"]
            ):
                facts.append((sentence, session_memory.importance_score))

        return facts[:5]  # Limit to 5 facts

    def _categorize_fact(self, fact: str) -> str:
        """Categorize a fact based on its content."""
        fact_lower = fact.lower()

        categories = {
            "preference": ["prefer", "like", "dislike", "want"],
            "relationship": ["know", "friend", "colleague", "reports to"],
            "project": ["project", "working on", "implementing"],
            "technical": ["code", "api", "database", "server"],
            "personal": ["live", "from", "called", "name"],
        }

        for category, keywords in categories.items():
            if any(keyword in fact_lower for keyword in keywords):
                return category

        return "general"

    async def shutdown(self) -> None:
        """Cleanup resources."""
        await self.longterm.shutdown()
