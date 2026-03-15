# Memory System Test Plan

## Overview
Comprehensive testing strategy for the enhanced memory system including unit tests, integration tests, performance benchmarks, and end-to-end testing.

---

## Test Environment Setup

### Test Infrastructure
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock
pip install pytest-benchmark faker-freezegun

# Setup test databases
mkdir -p tests/fixtures/memory
mkdir -p tests/temp/vector_db
```

### Test Configuration
```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def test_workspace(tmp_path):
    """Provide temporary workspace for tests."""
    workspace = tmp_path / "memory_test"
    workspace.mkdir()
    return workspace

@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider for unit tests."""
    # Return fixed embeddings for deterministic testing
    pass
```

---

## Phase 1: Unit Tests

### 1.1 Embedding Provider Tests
```python
# tests/test_embeddings.py

class TestEmbeddingProviders:
    def test_openai_provider_initialization(self):
        """Test OpenAI provider setup with valid API key."""
        pass

    def test_local_provider_initialization(self):
        """Test sentence-transformers provider setup."""
        pass

    def test_ollama_provider_initialization(self):
        """Test Ollama provider connection."""
        pass

    def test_embedding_generation(self):
        """Test single text embedding generation."""
        pass

    def test_batch_embedding_generation(self):
        """Test batch embedding with multiple texts."""
        pass

    def test_embedding_dimensions(self):
        """Verify embeddings have correct dimensions."""
        pass

    def test_embedding_caching(self):
        """Test that embeddings are cached properly."""
        pass

    def test_provider_fallback(self):
        """Test fallback from primary to secondary provider."""
        pass

    def test_error_handling(self):
        """Test handling of API errors and timeouts."""
        pass
```

### 1.2 Vector Store Tests
```python
# tests/test_vector_store.py

class TestVectorStore:
    def test_store_initialization(self):
        """Test ChromaDB/FAISS store creation."""
        pass

    def test_add_memory_entry(self):
        """Test adding a single memory entry."""
        pass

    def test_batch_add_entries(self):
        """Test adding multiple memory entries."""
        pass

    def test_delete_entry(self):
        """Test deleting a memory entry."""
        pass

    def test_update_entry(self):
        """Test updating an existing entry."""
        pass

    def test_search_by_vector(self):
        """Test vector similarity search."""
        pass

    def test_search_with_filters(self):
        """Test search with metadata filters."""
        pass

    def test_persistence(self):
        """Test that data persists across restarts."""
        pass

    def test_index_rebuilding(self):
        """Test rebuilding vector index."""
        pass

    def test_storage_size_tracking(self):
        """Test monitoring storage usage."""
        pass
```

### 1.3 Memory Entry Tests
```python
# tests/test_memory_entry.py

class TestMemoryEntry:
    def test_memory_entry_creation(self):
        """Test creating a valid memory entry."""
        pass

    def test_metadata_validation(self):
        """Test metadata schema validation."""
        pass

    def test_importance_scoring(self):
        """Test importance score calculation."""
        pass

    def test_tag_assignment(self):
        """Test tag management and retrieval."""
        pass

    def test_access_counting(self):
        """Test tracking access frequency."""
        pass

    def test_timestamp_management(self):
        """Test creation and update timestamps."""
        pass

    def test_serialization(self):
        """Test JSON serialization/deserialization."""
        pass
```

### 1.4 Text Chunking Tests
```python
# tests/test_chunkers.py

class TestTextChunkers:
    def test_fixed_size_chunking(self):
        """Test chunking by token count."""
        pass

    def test_semantic_chunking(self):
        """Test chunking by semantic boundaries."""
        pass

    def test_code_chunking(self):
        """Test code-specific chunking logic."""
        pass

    def test_chunk_overlap(self):
        """Test overlap between chunks."""
        pass

    def test_minimum_chunk_size(self):
        """Test handling of small texts."""
        pass

    def test_large_document_chunking(self):
        """Test chunking of large documents."""
        pass
```

### 1.5 Search Algorithms Tests
```python
# tests/test_search.py

class TestSearchAlgorithms:
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        pass

    def test_bm25_ranking(self):
        """Test BM25 keyword ranking."""
        pass

    def test_hybrid_fusion(self):
        """Test combining vector and keyword results."""
        pass

    def test_query_expansion(self):
        """Test query term expansion."""
        pass

    def test_result_ranking(self):
        """Test result ordering and scoring."""
        pass

    def test_diversity_sampling(self):
        """Test diverse result selection."""
        pass

    def test_temporal_weighting(self):
        """Test time-based result weighting."""
        pass
```

---

## Phase 2: Integration Tests

### 2.1 Memory Pipeline Tests
```python
# tests/test_memory_pipeline.py

class TestMemoryPipeline:
    @pytest.mark.asyncio
    async def test_full_consolidation_flow(self):
        """Test complete memory consolidation process."""
        pass

    @pytest.mark.asyncio
    async def test_embedding_generation_workflow(self):
        """Test embedding generation in pipeline."""
        pass

    @pytest.mark.asyncio
    async def test_storage_workflow(self):
        """Test storage in vector database."""
        pass

    @pytest.mark.asyncio
    async def test_indexing_workflow(self):
        """Test FTS index updating."""
        pass

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test recovery from pipeline failures."""
        pass
```

### 2.2 Search Integration Tests
```python
# tests/test_search_integration.py

class TestSearchIntegration:
    def test_hybrid_search_workflow(self):
        """Test complete hybrid search flow."""
        pass

    def test_search_with_context_injection(self):
        """Test memory retrieval and context building."""
        pass

    def test_search_with_filters(self):
        """Test search with multiple filters."""
        pass

    def test_search_performance_with_large_db(self):
        """Test search with 100K+ memories."""
        pass
```

### 2.3 Agent Integration Tests
```python
# tests/test_agent_integration.py

class TestAgentIntegration:
    @pytest.mark.asyncio
    async def test_memory_tool_invocation(self):
        """Test agent calling memory tools."""
        pass

    @pytest.mark.asyncio
    async def test_context_enhancement(self):
        """Test automatic memory retrieval."""
        pass

    @pytest.mark.asyncio
    async def test_memory_consolidation_trigger(self):
        """Test automatic consolidation trigger."""
        pass

    @pytest.mark.asyncio
    async def test_conversation_with_memory(self):
        """Test full conversation with memory system."""
        pass
```

---

## Phase 3: Performance Tests

### 3.1 Benchmark Tests
```python
# tests/test_benchmarks.py

class TestPerformance:
    def test_embedding_generation_speed(self):
        """Test: <50ms per embedding."""
        pass

    def test_search_latency_small_db(self):
        """Test: <50ms for 1K memories."""
        pass

    def test_search_latency_medium_db(self):
        """Test: <100ms for 10K memories."""
        pass

    def test_search_latency_large_db(self):
        """Test: <200ms for 100K memories."""
        pass

    def test_batch_indexing_speed(self):
        """Test: >100 memories/second indexed."""
        pass

    def test_storage_efficiency(self):
        """Test: <10MB per 1000 memories."""
        pass

    def test_concurrent_searches(self):
        """Test: Handle 100 concurrent searches."""
        pass
```

### 3.2 Load Tests
```python
# tests/test_load.py

class TestLoad:
    @pytest.mark.slow
    def test_continuous_indexing(self):
        """Test indexing 10K memories continuously."""
        pass

    @pytest.mark.slow
    def test_sustained_search_load(self):
        """Test 1000 searches over 10 minutes."""
        pass

    @pytest.mark.slow
    def test_memory_growth(self):
        """Test behavior with 1M+ memories."""
        pass

    @pytest.mark.slow
    def test_concurrent_writes(self):
        """Test 50 concurrent memory writes."""
        pass
```

### 3.3 Memory Tests
```python
# tests/test_memory_usage.py

class TestMemoryUsage:
    def test_embedding_cache_memory(self):
        """Test cache memory usage stays bounded."""
        pass

    def test_vector_db_memory(self):
        """Test vector database memory usage."""
        pass

    def test_memory_leak_detection(self):
        """Test for memory leaks over time."""
        pass
```

---

## Phase 4: End-to-End Tests

### 4.1 User Journey Tests
```python
# tests/test_e2e.py

class TestE2E:
    @pytest.mark.e2e
    def test_new_user_conversation(self):
        """Test conversation with no existing memory."""
        pass

    @pytest.mark.e2e
    def test_returning_user_conversation(self):
        """Test conversation with existing memory."""
        pass

    @pytest.mark.e2e
    def test_long_conversation(self):
        """Test conversation over multiple sessions."""
        pass

    @pytest.mark.e2e
    def test_multi_topic_conversation(self):
        """Test conversation switching topics."""
        pass

    @pytest.mark.e2e
    def test_memory_recall_accuracy(self):
        """Test accurate recall of past information."""
        pass
```

### 4.2 Error Recovery Tests
```python
# tests/test_error_recovery.py

class TestErrorRecovery:
    def test_vector_db_failure_recovery(self):
        """Test graceful degradation if vector DB fails."""
        pass

    def test_embedding_provider_failure(self):
        """Test fallback when provider fails."""
        pass

    def test_corrupted_index_recovery(self):
        """Test recovery from corrupted index."""
        pass

    def test_disk_space_handling(self):
        """Test behavior when disk is full."""
        pass
```

---

## Phase 5: Edge Cases & Error Handling

### 5.1 Edge Case Tests
```python
# tests/test_edge_cases.py

class TestEdgeCases:
    def test_empty_memory_database(self):
        """Test search with empty database."""
        pass

    def test_very_long_query(self):
        """Test handling of extremely long queries."""
        pass

    def test_special_characters(self):
        """Test handling of special characters."""
        pass

    def test_multilingual_content(self):
        """Test handling of non-English content."""
        pass

    def test_duplicate_detection(self):
        """Test detection of duplicate memories."""
        pass

    def test_concurrent_same_query(self):
        """Test concurrent identical queries."""
        pass

    def test_rapid_fire_queries(self):
        """Test handling many queries in quick succession."""
        pass
```

### 5.2 Security Tests
```python
# tests/test_security.py

class TestSecurity:
    def test_injection_prevention(self):
        """Test prevention of prompt injection."""
        pass

    def test_data_isolation(self):
        """Test memory isolation between users."""
        pass

    def test_sensitive_data_handling(self):
        """Test handling of sensitive information."""
        pass

    def test_access_control(self):
        """Test memory access permissions."""
        pass
```

---

## Test Data & Fixtures

### Sample Data
```python
# tests/fixtures/memory_data.py

SAMPLE_CONVERSATIONS = [
    {
        "role": "user",
        "content": "My favorite programming language is Python",
        "timestamp": "2024-01-01T10:00:00"
    },
    {
        "role": "assistant",
        "content": "That's great! Python is versatile and easy to learn.",
        "timestamp": "2024-01-01T10:00:05"
    }
]

SAMPLE_DOCUMENTS = [
    {
        "content": "Machine learning is a subset of artificial intelligence...",
        "source": "document",
        "tags": ["ml", "ai"]
    }
]

SAMPLE_CODE_SNIPPETS = [
    {
        "content": "def hello_world():\n    print('Hello, World!')",
        "language": "python",
        "source": "code"
    }
]
```

---

## Continuous Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/test-memory-system.yml
name: Memory System Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov

    - name: Run unit tests
      run: pytest tests/unit/ -v --cov=nanobot/agent/memory

    - name: Run integration tests
      run: pytest tests/integration/ -v

    - name: Run performance tests
      run: pytest tests/performance/ -v -m benchmark

    - name: Generate coverage report
      run: pytest --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## Test Execution

### Run All Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=nanobot/agent/memory --cov-report=html

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/performance/ -v -m benchmark
```

### Run Specific Tests
```bash
# Run embedding tests
pytest tests/test_embeddings.py -v

# Run search tests
pytest tests/test_search.py -v

# Run slow tests
pytest tests/ -v -m slow

# Run E2E tests
pytest tests/ -v -m e2e
```

---

## Success Criteria

### Test Coverage
- **Unit Tests**: >90% code coverage
- **Integration Tests**: >80% coverage of critical paths
- **End-to-End Tests**: All main user journeys covered

### Performance Benchmarks
- **Search Latency**: <100ms for 10K memories
- **Embedding Generation**: <50ms per text
- **Indexing Speed**: >100 memories/second
- **Memory Usage**: <500MB for 10K memories

### Quality Metrics
- **Test Pass Rate**: 100% (all tests must pass)
- **Flaky Test Rate**: <1%
- **Test Execution Time**: <5 minutes for full suite

---

## Test Maintenance

### Regular Updates
- Review and update tests monthly
- Add regression tests for discovered bugs
- Update test data as features evolve
- Monitor test execution times

### Documentation
- Document test cases in code comments
- Maintain test data documentation
- Track test coverage trends
- Report test metrics in CI/CD

---

## Debugging Failed Tests

### Common Issues
1. **Flaky Tests**: Add retries or increase timeouts
2. **Resource Leaks**: Ensure proper cleanup in fixtures
3. **Race Conditions**: Use proper synchronization
4. **Environment Issues**: Use containers for consistency

### Debugging Tools
```bash
# Run with verbose output
pytest tests/ -vv -s

# Run with debugger
pytest tests/ --pdb

# Run specific test with output
pytest tests/test_search.py::TestSearch::test_hybrid_search -vv
```

---

## Next Steps

1. Implement Phase 1 tests first (unit tests)
2. Add tests incrementally with features
3. Set up CI/CD pipeline
4. Monitor test results and coverage
5. Refactor and optimize based on test insights
