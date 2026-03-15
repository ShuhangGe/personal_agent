# 🧠 Memory System Architecture Plan for Nanobot

## Current State Analysis

### Nanobot's Current Memory
- Basic file-based storage (`MEMORY.md` + `HISTORY.md`)
- Simple consolidation via LLM summarization
- No semantic search or vector embeddings
- Limited retrieval capabilities

### OpenClaw's Memory System (Reference)
- SQLite database with vector extensions
- Full-text search (FTS) + vector embeddings
- Hybrid search (semantic + keyword)
- Multiple embedding providers (OpenAI, Gemini, Voyage, Mistral, Ollama)
- File watching and automatic indexing
- Batch processing and caching

---

## Implementation Plan

### Phase 1: Core Infrastructure (Foundation)

#### 1.1 Vector Database Integration
- **Choose Vector DB**: `ChromaDB` (Python-native, easy setup) or `FAISS` (lightweight)
- **SQLite + Vector Extension**: Use `sqlite-vss` or `chromadb-sqlite` for integrated storage
- **File System Structure**:
  ```
  workspace/
  ├── memory/
  │   ├── vector_db/          # Vector database storage
  │   ├── embeddings/         # Cached embeddings
  │   ├── MEMORY.md           # Long-term facts (existing)
  │   ├── HISTORY.md          # Conversation history (existing)
  │   └── index/              # FTS index
  ```

#### 1.2 Embedding Provider System
- **Providers**: OpenAI, local models (Ollama), sentence-transformers
- **Embedding models**:
  - Default: `text-embedding-3-small` (OpenAI, cheap & fast)
  - Local: `all-MiniLM-L6-v2` (sentence-transformers)
  - Fallback: `mxbai-embed-large` (Ollama)
- **Caching**: Cache embeddings to avoid re-computation

---

### Phase 2: Memory Storage & Indexing

#### 2.1 Enhanced Memory Schema
Create a unified memory structure:
```python
class MemoryEntry:
    id: str
    content: str
    embedding: Optional[np.ndarray]
    metadata: dict {
        timestamp: datetime
        source: str  # "conversation", "document", "code", etc.
        session_id: str
        importance: float  # 0-1 score
        tags: List[str]
        access_count: int
        last_accessed: datetime
    }
```

#### 2.2 File System Storage
- **Chunking Strategy**: Split documents into semantic chunks (512-1024 tokens)
- **Storage Format**: JSON + SQLite for metadata
- **Version Control**: Track memory changes over time

#### 2.3 Automatic Indexing
- **File Watcher**: Monitor workspace files for changes
- **Incremental Updates**: Only re-index changed content
- **Priority Queue**: Index important content first

---

### Phase 3: Search & Retrieval

#### 3.1 Hybrid Search System
```python
class MemorySearchEngine:
    def search(self, query: str, mode: str = "hybrid") -> List[MemoryEntry]:
        """
        Modes:
        - "vector": Pure semantic search
        - "keyword": Full-text search
        - "hybrid": Combined vector + keyword (default)
        - "temporal": Time-weighted recent memories
        """
```

#### 3.2 Search Algorithms
- **Vector Search**: Cosine similarity on embeddings
- **Keyword Search**: BM25 ranking
- **Hybrid Fusion**: Combine both with learned weights
- **Query Expansion**: Expand queries with related terms

#### 3.3 Memory Retrieval Strategies
- **Context Window Awareness**: Retrieve relevant memories based on available tokens
- **Temporal Decay**: Prioritize recent memories
- **Importance Scoring**: Factor in access frequency and manual importance
- **Diversity**: Ensure retrieved memories cover different topics

---

### Phase 4: Integration with Nanobot

#### 4.1 Agent Loop Integration
```python
class EnhancedMemoryConsolidator(MemoryConsolidator):
    async def consolidate_with_embeddings(self, messages: List[dict]) -> bool:
        # 1. Generate embeddings for new messages
        # 2. Store in vector database
        # 3. Update FTS index
        # 4. Consolidate into MEMORY.md (existing logic)
```

#### 4.2 Memory Tools for Agent
Add tools to the agent's toolkit:
```python
TOOLS = [
    {
        "name": "search_memory",
        "description": "Search memory using semantic and keyword search",
        "parameters": {
            "query": "string",
            "mode": "hybrid|vector|keyword",
            "limit": 5
        }
    },
    {
        "name": "store_memory",
        "description": "Store important information in long-term memory",
        "parameters": {
            "content": "string",
            "importance": "float",
            "tags": ["list"]
        }
    }
]
```

#### 4.3 Context Enhancement
- **Automatic Memory Retrieval**: Fetch relevant memories before processing user messages
- **Memory Injection**: Inject retrieved memories into system prompt
- **Dynamic Context**: Adjust context based on conversation relevance

---

### Phase 5: Advanced Features

#### 5.1 Memory Management
- **Deduplication**: Detect and merge similar memories
- **Memory Pruning**: Remove outdated or low-importance memories
- **Memory Summarization**: Periodically summarize old memories

#### 5.2 Multi-Modal Memory
- **Code Memory**: Specialized handling for code snippets
- **Document Memory**: Parse and index PDFs, docs, etc.
- **Conversation Memory**: Better handling of dialogue history

#### 5.3 Memory Analytics
- **Access Patterns**: Track which memories are most useful
- **Memory Quality Score**: Rate memory helpfulness
- **Storage Optimization**: Compress and archive old memories

---

## Technical Stack

### Core Dependencies
```python
# Vector Database
chromadb>=0.4.0          # Primary vector DB
# OR
faiss-cpu>=1.7.0         # Alternative: FAISS

# Embeddings
openai>=1.0.0            # OpenAI embeddings
sentence-transformers     # Local embeddings
ollama>=0.1.0            # Ollama integration

# Search & Storage
sqlite3                  # Built-in
whoosh>=2.7.4            # Full-text search (alternative)
sqlite-vss>=0.1.0        # SQLite vector extension

# File Processing
watchdog>=3.0.0          # File watching
python-magic>=0.4.27     # File type detection
```

---

## File Structure
```
nanobot/
├── agent/
│   ├── memory/              # NEW: Enhanced memory system
│   │   ├── __init__.py
│   │   ├── vector_store.py  # Vector database management
│   │   ├── embeddings.py    # Embedding providers
│   │   ├── search.py        # Search algorithms
│   │   ├── indexing.py      # File indexing
│   │   ├── retrieval.py     # Memory retrieval strategies
│   │   └── memory.py        # Enhanced memory (existing)
├── utils/
│   ├── chunkers.py          # NEW: Text chunking strategies
│   └── deduplication.py     # NEW: Memory deduplication
```

---

## Implementation Order

### Week 1: Core Infrastructure
- Setup vector database (ChromaDB)
- Implement embedding providers (OpenAI, local)
- Create memory schema and data structures

### Week 2: Storage & Indexing
- Implement memory storage system
- Build file indexing pipeline
- Add embedding caching

### Week 3: Search & Retrieval
- Implement vector search
- Add keyword search (BM25)
- Build hybrid search fusion

### Week 4: Integration
- Integrate with nanobot agent loop
- Add memory tools
- Implement context enhancement

### Week 5: Advanced Features
- Add memory deduplication
- Implement memory pruning
- Build analytics dashboard

---

## Success Metrics

- **Search Accuracy**: Relevant memories in top 5 results (>80%)
- **Performance**: Search latency <100ms for 10K memories
- **Storage Efficiency**: <1GB for 100K memories
- **Agent Enhancement**: Improved context awareness and response quality
- **User Satisfaction**: Better conversation continuity and personalization

---

## Configuration

### Memory Settings
```yaml
memory:
  enabled: true
  vector_db:
    type: "chromadb"  # or "faiss"
    path: "workspace/memory/vector_db"
  embeddings:
    provider: "openai"  # "openai", "local", "ollama"
    model: "text-embedding-3-small"
    dimension: 1536
    cache_size: 1000
  search:
    default_mode: "hybrid"
    max_results: 10
    similarity_threshold: 0.7
  indexing:
    auto_index: true
    chunk_size: 512
    chunk_overlap: 50
```

---

## Migration Path

### From Current Memory System
1. Keep existing `MEMORY.md` and `HISTORY.md` files
2. Migrate historical content to vector database
3. Maintain backward compatibility during transition
4. Gradually phase out old system

### Rollback Strategy
- Keep old memory system as fallback
- Feature flags to enable/disable new system
- Graceful degradation if vector DB fails
