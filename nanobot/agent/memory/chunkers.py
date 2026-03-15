"""Text chunking utilities for processing large documents."""

from __future__ import annotations

import re
from typing import list
from dataclasses import dataclass

from loguru import logger


@dataclass
class TextChunk:
    """A chunk of text with metadata."""

    content: str
    index: int
    total: int
    start_pos: int
    end_pos: int


class FixedSizeChunker:
    """Chunk text into fixed-size segments."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[TextChunk]:
        """Chunk text into fixed-size segments.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [TextChunk(
                content=text,
                index=0,
                total=1,
                start_pos=0,
                end_pos=len(text),
            )]

        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append(TextChunk(
                content=chunk_text,
                index=index,
                total=0,  # Will update after
                start_pos=start,
                end_pos=min(end, len(text)),
            ))

            index += 1
            start = end - self.overlap

        # Update total count
        for chunk in chunks:
            chunk.total = len(chunks)

        logger.debug(f"Chunked text into {len(chunks)} chunks")
        return chunks


class SemanticChunker:
    """Chunk text based on semantic boundaries (paragraphs, sentences)."""

    def __init__(self, max_chunk_size: int = 1024, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[TextChunk]:
        """Chunk text based on semantic boundaries.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [TextChunk(
                content=text,
                index=0,
                total=1,
                start_pos=0,
                end_pos=len(text),
            )]

        chunks = []
        current_chunk = ""
        start_pos = 0
        index = 0

        for i, paragraph in enumerate(paragraphs):
            # Check if adding this paragraph would exceed max size
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(TextChunk(
                    content=current_chunk.strip(),
                    index=index,
                    total=0,  # Will update after
                    start_pos=start_pos,
                    end_pos=start_pos + len(current_chunk),
                ))

                index += 1
                start_pos += len(current_chunk) - self.overlap
                current_chunk = paragraph

                # Add overlap from previous chunk
                if chunks and self.overlap > 0:
                    prev_chunk = chunks[-1].content
                    overlap_text = prev_chunk[-self.overlap:] if len(prev_chunk) > self.overlap else prev_chunk
                    current_chunk = overlap_text + " " + current_chunk
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk:
            chunks.append(TextChunk(
                content=current_chunk.strip(),
                index=index,
                total=0,
                start_pos=start_pos,
                end_pos=start_pos + len(current_chunk),
            ))

        # Update total count
        for chunk in chunks:
            chunk.total = len(chunks)

        logger.debug(f"Semantically chunked text into {len(chunks)} chunks")
        return chunks


class CodeChunker:
    """Chunk code files while preserving structure."""

    def __init__(self, max_chunk_size: int = 1024, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk(self, code: str, language: str = "python") -> list[TextChunk]:
        """Chunk code while preserving function/class structure.

        Args:
            code: Code to chunk
            language: Programming language

        Returns:
            List of code chunks
        """
        if language == "python":
            return self._chunk_python(code)
        else:
            # Fallback to fixed-size chunking
            return FixedSizeChunker(self.max_chunk_size, self.overlap).chunk(code)

    def _chunk_python(self, code: str) -> list[TextChunk]:
        """Chunk Python code by functions and classes."""
        # Simple Python-specific chunking
        # In production, would use ast.parse for better accuracy

        chunks = []
        lines = code.split('\n')
        current_chunk = []
        current_size = 0
        start_pos = 0
        index = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline

            # Check for function/class definition
            if line.strip().startswith(('def ', 'class ')):
                # Start new chunk if current is large enough
                if current_size > self.max_chunk_size // 2:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(TextChunk(
                        content=chunk_text,
                        index=index,
                        total=0,
                        start_pos=start_pos,
                        end_pos=start_pos + len(chunk_text),
                    ))
                    index += 1
                    start_pos += len(chunk_text)
                    current_chunk = []
                    current_size = 0

            current_chunk.append(line)
            current_size += line_size

            # Check if chunk is too large
            if current_size > self.max_chunk_size:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(TextChunk(
                    content=chunk_text,
                    index=index,
                    total=0,
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_text),
                ))
                index += 1
                start_pos += len(chunk_text)
                current_chunk = []
                current_size = 0

        # Add remaining code
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(TextChunk(
                content=chunk_text,
                index=index,
                total=0,
                start_pos=start_pos,
                end_pos=start_pos + len(chunk_text),
            ))

        # Update total count
        for chunk in chunks:
            chunk.total = len(chunks)

        logger.debug(f"Chunked Python code into {len(chunks)} chunks")
        return chunks


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    # Rough estimation: ~4 characters per token
    return len(text) // 4


def get_chunker(
    content_type: str = "text",
    chunk_size: int = 512,
    overlap: int = 50,
) -> FixedSizeChunker | SemanticChunker | CodeChunker:
    """Get appropriate chunker based on content type.

    Args:
        content_type: Type of content ("text", "code", "semantic")
        chunk_size: Maximum chunk size
        overlap: Overlap between chunks

    Returns:
        Appropriate chunker instance
    """
    if content_type == "code":
        return CodeChunker(max_chunk_size=chunk_size, overlap=overlap)
    elif content_type == "semantic":
        return SemanticChunker(max_chunk_size=chunk_size, overlap=overlap)
    else:
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
