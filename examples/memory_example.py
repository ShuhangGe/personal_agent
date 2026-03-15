"""Example usage of the enhanced memory system."""

import asyncio
from pathlib import Path

from nanobot.agent.enhanced_memory import (
    EnhancedMemorySystem,
    MemorySource,
    SearchMode,
)


async def main():
    """Demonstrate enhanced memory system capabilities."""

    # Initialize workspace
    workspace = Path("./memory_test")
    workspace.mkdir(exist_ok=True)

    # Create enhanced memory system
    memory_system = EnhancedMemorySystem(
        workspace=workspace,
        embedding_provider_type="local",  # Use local embeddings
        enable_vector_search=True,
        enable_keyword_search=True,
    )

    try:
        # Initialize the system
        await memory_system.initialize()
        print("✅ Enhanced memory system initialized")

        # Add some sample memories
        print("\n📝 Adding sample memories...")

        await memory_system.add_memory(
            content="Python is my favorite programming language",
            source=MemorySource.CONVERSATION,
            session_id="session1",
            importance=0.8,
            tags=["programming", "python"],
        )

        await memory_system.add_memory(
            content="I love working with machine learning and AI",
            source=MemorySource.CONVERSATION,
            session_id="session1",
            importance=0.7,
            tags=["ai", "ml"],
        )

        await memory_system.add_memory(
            content="def hello_world():\n    print('Hello, World!')",
            source=MemorySource.CODE,
            session_id="session1",
            importance=0.5,
            tags=["code", "python"],
        )

        print("✅ Added 3 sample memories")

        # Search for memories
        print("\n🔍 Searching for memories...")

        # Vector search
        result = await memory_system.search_memory(
            query="What programming languages do I like?",
            mode=SearchMode.VECTOR,
            max_results=5,
            similarity_threshold=0.0,  # Show all results
        )

        print(f"\n📊 Vector search results ({result.total_results} found):")
        for i, res in enumerate(result.get_top_results(3), 1):
            print(f"{i}. [{res.score:.2f}] {res.memory.content[:100]}...")

        # Hybrid search
        result = await memory_system.search_memory(
            query="programming code",
            mode=SearchMode.HYBRID,
            max_results=5,
            similarity_threshold=0.0,  # Show all results
        )

        print(f"\n📊 Hybrid search results ({result.total_results} found):")
        for i, res in enumerate(result.get_top_results(3), 1):
            print(f"{i}. [{res.score:.2f}] {res.memory.content[:100]}...")

        # Get context for prompt
        print("\n💬 Getting context for prompt injection...")
        context = await memory_system.get_context_for_prompt(
            query="Tell me about my programming preferences",
            max_tokens=500,
        )
        print("Context:")
        print(context)

        # Get statistics
        print("\n📈 Memory system statistics:")
        stats = await memory_system.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

    finally:
        # Clean up
        await memory_system.close()
        print("\n✅ Memory system closed")


if __name__ == "__main__":
    asyncio.run(main())
