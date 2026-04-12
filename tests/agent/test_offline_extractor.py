"""Tests for agent/offline_extractor.py — offline memory extraction and compression.

Integration tests load the actual GLiNER2 and LLMLingua-2 models and process
real text.  The only mock is a guard on call_llm to guarantee no LLM calls
leak through.

The ``TestRecursiveSplit`` class tests the chunking algorithm in isolation
using only the tokenizer (no model inference), so it runs without the
``integration`` marker.

Requires: pip install gliner2 llmlingua
"""

import json

import pytest
from unittest.mock import patch

from agent.offline_extractor import (
    extract_memories, compress_summary,
    _recursive_split, _MAX_CHUNK_TOKENS, _SEPARATORS,
)
from agent.context_compressor import SUMMARY_PREFIX


# ── Guard: fail hard if any LLM call is attempted ──────────────────────────

def _llm_call_guard(*args, **kwargs):
    raise AssertionError(
        "LLM call attempted during offline-only test — "
        "the offline strategy must not invoke call_llm"
    )


# ── Sample conversation data ───────────────────────────────────────────────

_SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I prefer using Python 3.12 with type hints for all my projects."},
    {"role": "assistant", "content": "Noted! I'll use Python 3.12 with type hints."},
    {"role": "user", "content": "My development setup is NeoVim on Arch Linux with Kitty terminal."},
    {"role": "assistant", "content": "Great setup. I'll keep that in mind."},
    {"role": "user", "content": "We decided to use FastAPI instead of Flask for the new microservice."},
    {"role": "assistant", "content": "FastAPI it is. I'll structure the project accordingly."},
    {"role": "user", "content": "Don't ever use wildcard imports in this codebase, they cause namespace pollution."},
    {"role": "assistant", "content": "Understood, explicit imports only."},
]

_SAMPLE_SERIALIZED = """[USER]: I prefer using Python 3.12 with type hints for all my projects.

[ASSISTANT]: Noted! I'll use Python 3.12 with type hints.

[USER]: My development setup is NeoVim on Arch Linux with Kitty terminal.

[ASSISTANT]: Great setup. I'll keep that in mind.

[USER]: We decided to use FastAPI instead of Flask for the new microservice.

[ASSISTANT]: FastAPI it is. I'll structure the project accordingly.

[USER]: Don't ever use wildcard imports in this codebase, they cause namespace pollution.

[ASSISTANT]: Understood, explicit imports only.

[USER]: The database migration failed on the staging server with error code PG-4012. We need to fix the foreign key constraint on the users table before proceeding with the deployment.

[ASSISTANT]: I'll investigate the foreign key constraint issue on the users table. Let me look at the migration file."""


# ── Test: offline memory extraction (GLiNER2) ──────────────────────────────

@pytest.mark.integration
class TestOfflineMemoryExtraction:
    """Verify extract_memories produces tool_calls compatible with the
    built-in memory tool, using real GLiNER2 inference."""

    def test_extract_memories_produces_valid_tool_calls(self):
        with patch("agent.auxiliary_client.call_llm", side_effect=_llm_call_guard):
            tool_calls = extract_memories(_SAMPLE_MESSAGES)

        # Must return a list (may be empty if model doesn't fire, but
        # with this input it should find at least one entity).
        assert isinstance(tool_calls, list)

        for tc in tool_calls:
            # Structural contract: matches what flush_memories iterates over
            assert hasattr(tc, "function"), "tool_call must have .function"
            assert hasattr(tc.function, "name"), "tool_call.function must have .name"
            assert hasattr(tc.function, "arguments"), "tool_call.function must have .arguments"

            assert tc.function.name == "memory"

            # Arguments must be valid JSON parseable by json.loads
            args = json.loads(tc.function.arguments)
            assert isinstance(args, dict)

            # Required fields for memory_tool()
            assert "action" in args, "arguments must include 'action'"
            assert "target" in args, "arguments must include 'target'"
            assert "content" in args, "arguments must include 'content'"

            # Value constraints matching memory_tool validation
            assert args["action"] == "add", "offline extraction should only add entries"
            assert args["target"] in ("memory", "user"), \
                f"target must be 'memory' or 'user', got '{args['target']}'"
            assert isinstance(args["content"], str) and len(args["content"]) >= 15, \
                "content must be a non-trivial string (>=15 chars)"

    def test_extract_memories_respects_max_cap(self):
        """Output must not exceed _MAX_MEMORIES to stay within MEMORY.md budget."""
        from agent.offline_extractor import _MAX_MEMORIES

        with patch("agent.auxiliary_client.call_llm", side_effect=_llm_call_guard):
            tool_calls = extract_memories(_SAMPLE_MESSAGES)

        assert len(tool_calls) <= _MAX_MEMORIES


# ── Test: offline context compression (LLMLingua-2) ────────────────────────

@pytest.mark.integration
class TestOfflineCompression:
    """Verify compress_summary produces output compatible with the
    ContextCompressor summary pipeline, using real LLMLingua-2 inference."""

    def test_compress_summary_produces_valid_output(self):
        with patch("agent.auxiliary_client.call_llm", side_effect=_llm_call_guard):
            result = compress_summary(_SAMPLE_SERIALIZED)

        assert result is not None, "compress_summary must return a string, not None"
        assert isinstance(result, str)
        assert len(result) > 0, "compressed output must be non-empty"

        ratio = len(result) / len(_SAMPLE_SERIALIZED)
        assert ratio < 0.6, (
            f"compression must achieve at least 40% reduction, "
            f"got {ratio:.1%} ({len(result)}/{len(_SAMPLE_SERIALIZED)} chars)"
        )

    def test_compressed_output_compatible_with_summary_prefix(self):
        """The output must be usable with ContextCompressor._with_summary_prefix
        which prepends SUMMARY_PREFIX to create the final compaction message."""
        with patch("agent.auxiliary_client.call_llm", side_effect=_llm_call_guard):
            result = compress_summary(_SAMPLE_SERIALIZED)

        # Simulate what _generate_summary does with the result
        prefixed = f"{SUMMARY_PREFIX}\n{result}"
        assert prefixed.startswith(SUMMARY_PREFIX)
        # The combined output must be a valid string the model can read
        assert len(prefixed) > len(SUMMARY_PREFIX)

    def test_custom_rate_produces_larger_output(self):
        """A higher offline_rate should retain more content."""
        with patch("agent.auxiliary_client.call_llm", side_effect=_llm_call_guard):
            result_30 = compress_summary(_SAMPLE_SERIALIZED, rate=0.3)
            result_60 = compress_summary(_SAMPLE_SERIALIZED, rate=0.6)

        assert len(result_60) > len(result_30), \
            "rate=0.6 must produce more output than rate=0.3"


# ── Test: recursive token-aware splitter ────────────────────────────────────

@pytest.fixture(scope="module")
def tokenizer():
    """Load only the tokenizer (no full model) for fast splitter tests."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank")


class TestRecursiveSplit:
    """Corner-case tests for _recursive_split.  These run without the
    ``integration`` marker — they only need the tokenizer, not the models."""

    def _tok_len(self, tokenizer, text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    # ── Basic guarantees ────────────────────────────────────────────

    def test_short_text_returns_single_chunk(self, tokenizer):
        text = "Hello world."
        chunks = _recursive_split(tokenizer, text, _SEPARATORS, _MAX_CHUNK_TOKENS)
        assert chunks == [text]

    def test_empty_text_returns_empty(self, tokenizer):
        assert _recursive_split(tokenizer, "", _SEPARATORS, _MAX_CHUNK_TOKENS) == []

    def test_whitespace_only_returns_empty(self, tokenizer):
        assert _recursive_split(tokenizer, "   \n\n  \n  ", _SEPARATORS, _MAX_CHUNK_TOKENS) == []

    # ── Token limit enforcement ─────────────────────────────────────

    def test_every_chunk_within_token_limit(self, tokenizer):
        """No chunk may exceed _MAX_CHUNK_TOKENS regardless of input."""
        # Build text with mixed separators: paragraphs, lines, sentences
        text = "\n\n".join([
            "Short paragraph.",
            "Another short one.",
            "A. " * 300,  # ~300 tokens, single paragraph of sentences
            "\n".join(["line " * 40] * 20),  # long lines within a paragraph
            "word " * 600,  # single block with no sentence boundaries
        ])
        chunks = _recursive_split(tokenizer, text, _SEPARATORS, _MAX_CHUNK_TOKENS)
        for i, chunk in enumerate(chunks):
            t = self._tok_len(tokenizer, chunk)
            assert t <= _MAX_CHUNK_TOKENS, \
                f"chunk {i} has {t} tokens (limit {_MAX_CHUNK_TOKENS})"

    def test_custom_limit_respected(self, tokenizer):
        """Verify a small custom limit works correctly."""
        text = "word " * 100  # ~100 tokens
        limit = 20
        chunks = _recursive_split(tokenizer, text, _SEPARATORS, limit)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            t = self._tok_len(tokenizer, chunk)
            assert t <= limit, f"chunk {i} has {t} tokens (limit {limit})"

    # ── Lossless round-trip ─────────────────────────────────────────

    def test_no_content_words_lost(self, tokenizer):
        """Every word in the input must appear in exactly one chunk."""
        text = "\n\n".join([
            "[USER]: I prefer Python 3.12 with type hints.",
            "[ASSISTANT]: Noted!",
            "[USER]: " + "important detail. " * 200,
            "[ASSISTANT]: " + "response content. " * 200,
        ])
        chunks = _recursive_split(tokenizer, text, _SEPARATORS, _MAX_CHUNK_TOKENS)
        original_words = set(text.split())
        reassembled_words = set("\n\n".join(chunks).split())
        lost = original_words - reassembled_words
        assert not lost, f"lost {len(lost)} words: {list(lost)[:10]}"

    # ── Separator hierarchy ─────────────────────────────────────────

    def test_prefers_paragraph_boundaries(self, tokenizer):
        """When paragraphs fit, should not split within them."""
        p1 = "First paragraph with some content."
        p2 = "Second paragraph with other content."
        text = f"{p1}\n\n{p2}"
        chunks = _recursive_split(tokenizer, text, _SEPARATORS, _MAX_CHUNK_TOKENS)
        assert chunks == [text], "short paragraphs should stay in one chunk"

    def test_falls_back_to_line_split(self, tokenizer):
        """An oversized paragraph should split on line boundaries."""
        lines = [f"Line {i}: " + "content " * 30 for i in range(20)]
        text = "\n".join(lines)  # single paragraph, many lines
        chunks = _recursive_split(tokenizer, text, _SEPARATORS, _MAX_CHUNK_TOKENS)
        assert len(chunks) > 1
        # Lines should not be split mid-line
        for chunk in chunks:
            for line in chunk.split("\n"):
                assert line in lines or not line.strip(), \
                    f"line was split mid-content: {line[:60]!r}"

    def test_falls_back_to_sentence_split(self, tokenizer):
        """A single long line should split on sentence boundaries."""
        sentences = [f"Sentence number {i} with some detail." for i in range(80)]
        text = " ".join(sentences)  # one line, many sentences
        chunks = _recursive_split(tokenizer, text, _SEPARATORS, _MAX_CHUNK_TOKENS)
        assert len(chunks) > 1

    def test_hard_splits_on_no_separators(self, tokenizer):
        """A single unbreakable token sequence must hard-split by token ids."""
        text = "x" * 5000  # one long "word", no spaces/newlines/periods
        chunks = _recursive_split(tokenizer, text, _SEPARATORS, _MAX_CHUNK_TOKENS)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            t = self._tok_len(tokenizer, chunk)
            assert t <= _MAX_CHUNK_TOKENS, \
                f"hard-split chunk {i} has {t} tokens"

    def test_no_infinite_recursion_when_separators_absent(self, tokenizer):
        """Text with none of the hierarchy separators must terminate via
        hard-split.  Recursion is bounded by len(_SEPARATORS) + 1 = 5."""
        # No \n\n, no \n, no ". ", no " " — just one continuous token stream.
        text = "abcde" * 1000
        chunks = _recursive_split(tokenizer, text, _SEPARATORS, 100)
        assert len(chunks) > 1
        assert all(self._tok_len(tokenizer, c) <= 100 for c in chunks)
        # Verify all content survived the hard-split round-trip
        reassembled = "".join(chunks)
        assert reassembled.replace(" ", "") == text.replace(" ", "")

    def test_separator_that_does_not_split_still_terminates(self, tokenizer):
        """If a separator exists in the hierarchy but not in the text,
        split() returns [text] and recursion must advance to the next sep."""
        text = "word " * 200  # has " " but not \n\n, \n, or ". "
        chunks = _recursive_split(tokenizer, text, _SEPARATORS, 50)
        assert len(chunks) > 1
        assert all(self._tok_len(tokenizer, c) <= 50 for c in chunks)
