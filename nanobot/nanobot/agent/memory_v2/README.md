# Three-Tier Memory System

A sophisticated memory system for personal_agent inspired by OpenClaw's architecture, featuring semantic search, entity extraction, and intelligent consolidation.

## Features

### 🧠 Three-Tier Architecture

1. **Working Memory** (Short-term)
   - In-memory storage for active conversations
   - Fast access to recent context
   - Automatic cleanup of old sessions

2. **Session Memory** (Medium-term)
   - SQLite-backed storage with FTS5 full-text search
   - Vector embeddings for semantic search
   - Automatic expiration (configurable retention)
   - Entity extraction and topic classification

3. **Long-term Memory** (Persistent)
   - Hybrid Markdown + SQLite storage
   - Human-readable MEMORY.md file
   - Semantic search over facts
   - Import/export capabilities

### 🔍 Intelligent Search

- **Hybrid Search**: Combines keyword (BM25) and semantic (vector) search
- **Cross-tier**: Search across all memory tiers simultaneously
- **Ranked Results**: Reciprocal rank fusion for relevance
- **Filters**: Time range, category, importance score

### 🤖 Entity Extraction

- Named Entity Recognition (spaCy)
- Relationship extraction between entities
- Entity normalization and deduplication
- Confidence scoring

### 📊 Memory Consolidation

- Automatic consolidation based on token/time thresholds
- Importance scoring for selective promotion
- LLM-powered summarization (optional)
- Duplicate detection and merging

## Installation

### Basic Installation

```bash
conda activate open_manus
pip install -r requirements-memory.txt
```

### With spaCy (recommended for entity extraction)

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### With OpenAI embeddings (optional, faster)

```bash
pip install openai
export OPENAI_API_KEY="your-api-key"
```

## Quick Start

### Basic Usage

```python
from nanobot.agent.memory_v2 import MemoryManager, MemoryTier
from pathlib import Path

# Initialize memory manager
manager = MemoryManager(workspace=Path("."))
await manager.initialize()

# Store in working memory
manager.add_to_working_memory(
    content="User prefers dark mode",
    session_key="chat_123",
    role="user"
)

# Store in session memory
await manager.remember(
    content="User is working on a Python project",
    tier=MemoryTier.SESSION,
    importance=0.7
)

# Store in long-term memory
await manager.remember(
    content="User lives in San Francisco",
    tier=MemoryTier.LONGTERM,
    metadata={"category": "profile"},
    importance=1.0
)

# Search across all tiers
results = await manager.search("Python")
for result in results:
    print(f"[{result.tier.value}] {result.content}")

# Cleanup
await manager.shutdown()
```

### Configuration

Create a `config/memory.yaml` file:

```yaml
memory:
  tiers:
    working:
      max_messages: 100
      timeout_minutes: 30

    session:
      retention_days: 30
      consolidation_interval: 10

  embeddings:
    provider: "local"  # or "openai"
    model: "all-MiniLM-L6-v2"

  entities:
    enabled: true
```

Then load it:

```python
from nanobot.agent.memory_v2.config import MemoryConfig

config = MemoryConfig.from_yaml("config/memory.yaml")
manager = MemoryManager(workspace=Path("."), config=config)
```

## Architecture

### Data Flow

```
User Message
    ↓
Working Memory (in-memory)
    ↓ [Trigger: token/time limit]
Session Memory (SQLite + Vectors)
    ↓ [Trigger: importance threshold]
Long-term Memory (Markdown + SQLite)
```

### File Structure

```
memory/
├── MEMORY.md              # Long-term facts (human-readable)
├── HISTORY.md             # Event log (append-only)
├── session_memories.db    # Session memories + FTS5
└── longterm_memories.db   # Long-term facts index
```

## Advanced Usage

### Semantic Search

```python
# Search with semantic similarity
results = await manager.search(
    query="machine learning frameworks",
    tiers=[MemoryTier.SESSION, MemoryTier.LONGTERM],
    limit=10
)

# Results are ranked by relevance score
for result in results:
    print(f"[{result.score:.2f}] {result.content}")
```

### Filtered Search

```python
from nanobot.agent.memory_v2.models import SearchFilters
from datetime import datetime, timedelta

# Search recent memories
filters = SearchFilters(
    time_range=TimeRange(
        start=datetime.now() - timedelta(hours=24)
    ),
    min_importance=0.7
)

results = await manager.search(
    query="project",
    filters=filters,
    limit=10
)
```

### Memory Context for LLM

```python
# Get relevant memories for a query
context = await manager.get_memory_context(
    current_query="What do you know about the user?"
)

# Use in LLM prompt
system_prompt = f"""
You are an AI assistant with access to relevant context:

{context}

Answer the user's question based on this context.
"""
```

### Entity Extraction

```python
from nanobot.agent.memory_v2.entities import EntityExtractor

extractor = EntityExtractor(enabled=True)

text = "Alice works at Google in San Francisco"
entities = extractor.extract(text)

for entity in entities:
    print(f"{entity.name} - {entity.type}")
```

### Memory Consolidation

```python
from nanobot.agent.memory_v2.consolidator import MemoryConsolidator

consolidator = MemoryConsolidator(
    workspace=Path("."),
    embedder=manager.embedder,
    entity_extractor=manager.entity_extractor
)

await consolidator.initialize()

# Consolidate working memory to session
session_memory = await consolidator.consolidate_to_session(
    session_key="chat_123",
    working_memory=working_memory
)

# Consolidate to long-term if important
if session_memory.importance_score >= 0.7:
    await consolidator.consolidate_to_longterm(session_memory)
```

## API Reference

### MemoryManager

Main interface for memory operations.

**Methods:**
- `await initialize()` - Initialize database connections
- `await shutdown()` - Cleanup resources
- `await search(query, tiers=None, filters=None, limit=10)` - Search memories
- `await remember(content, tier, metadata=None, importance=0.5)` - Store memory
- `await forget(memory_id, tier)` - Remove memory
- `get_working_memory(session_key)` - Get working memory for session
- `add_to_working_memory(content, session_key, role="user")` - Add to working memory
- `clear_working_memory(session_key)` - Clear working memory
- `await get_stats(tier=None)` - Get memory statistics
- `await get_memory_context(current_query=None)` - Get relevant context

### MemoryTier

Enum for memory tiers:
- `MemoryTier.WORKING` - Short-term in-memory
- `MemoryTier.SESSION` - Medium-term SQLite
- `MemoryTier.LONGTERM` - Long-term persistent

### SearchFilters

Filters for memory search:
- `time_range` - Time range filter
- `categories` - Category filter
- `min_importance` - Minimum importance score
- `session_keys` - Specific sessions

## Configuration Options

### Working Memory
- `max_messages` - Maximum messages per session (default: 100)
- `max_tokens` - Maximum tokens per session (default: 4000)
- `timeout_minutes` - Session timeout (default: 30)

### Session Memory
- `retention_days` - Days to keep memories (default: 30)
- `max_memories` - Maximum memories stored (default: 10000)
- `consolidation_interval` - Messages before consolidation (default: 10)
- `importance_threshold` - Score for long-term promotion (default: 0.6)

### Long-term Memory
- `storage_dir` - Directory for memory files (default: "memory")
- `memory_file` - Long-term facts file (default: "MEMORY.md")
- `history_file` - Event log file (default: "HISTORY.md")
- `auto_consolidate` - Automatic consolidation (default: true)

### Embeddings
- `provider` - Embedding provider (default: "local")
- `model` - Model name (default: "all-MiniLM-L6-v2")
- `dimension` - Embedding dimension (default: 384)
- `batch_size` - Batch size for embeddings (default: 10)

### Entities
- `enabled` - Enable entity extraction (default: true)
- `min_confidence` - Minimum confidence (default: 0.7)
- `types` - Entity types to extract (default: [PERSON, ORG, GPE, ...])

## Performance

### Search Speed
- Working Memory: <1ms (in-memory)
- Session Memory: 10-50ms (SQLite + vectors)
- Long-term Memory: 20-100ms (file I/O + vectors)

### Storage Size
- ~20 MB per 1000 memories
- Includes embeddings, metadata, and indexes

### Recommendations
- Use local embeddings for privacy and cost savings
- Use OpenAI embeddings for speed and accuracy
- Enable Redis caching for multi-instance deployments
- Set appropriate retention policies to manage storage

## Troubleshooting

### Import Errors

```bash
# If you get "No module named 'sentence_transformers'"
pip install sentence-transformers

# If you get "No module named 'spacy'"
pip install spacy
python -m spacy download en_core_web_sm
```

### Database Locked

```python
# Enable WAL mode for better concurrency
await self._db.execute("PRAGMA journal_mode=WAL")
```

### Slow Embedding Generation

```python
# Reduce batch size or use GPU
config.embeddings.batch_size = 5  # Default: 10

# Or use OpenAI for faster embeddings
config.embeddings.provider = "openai"
```

## Examples

See `examples/memory_demo.py` for comprehensive examples including:
- Basic memory operations
- Entity extraction
- Memory consolidation
- Cross-tier search
- Memory statistics

Run the demo:

```bash
python examples/memory_demo.py
```

## Testing

```bash
# Run all tests
pytest personal_agent/nanobot/nanobot/agent/memory_v2/tests/

# Run with coverage
pytest --cov=personal_agent/nanobot/nanobot/agent/memory_v2

# Run specific test
pytest personal_agent/nanobot/nanobot/agent/memory_v2/tests/test_memory.py::TestWorkingMemory
```

## Contributing

The memory system is organized into several modules:

- `models.py` - Data structures and enums
- `config.py` - Configuration management
- `working.py` - Working memory store
- `session.py` - Session memory store
- `longterm.py` - Long-term memory store
- `embeddings.py` - Embedding provider
- `entities.py` - Entity extraction
- `consolidator.py` - Memory consolidation
- `manager.py` - Unified interface

## License

MIT License - See parent project LICENSE file.

## Credits

Inspired by [OpenClaw](https://github.com/openclaw/openclaw)'s memory architecture.
