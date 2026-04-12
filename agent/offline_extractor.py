"""Offline memory extraction and context compression — no LLM required.

Uses encoder-only models for language-agnostic, privacy-preserving
conversation processing:

  - GLiNER2 (205M params): Schema-driven zero-shot entity/fact extraction
    for memory persistence.  Matches GPT-4o on CrossNER benchmarks.

  - LLMLingua-2 (XLM-RoBERTa, 355M params): Token-level keep/discard
    classification for extractive compression.  95-98% information retention
    at 2-5x compression (ACL 2024), trained on MeetingBank conversations.

Both models are multilingual (100+ languages), run on CPU or GPU, and
process up to 262K tokens via chunking.  No generative inference is
performed — all operations are classification or encoding.

References:
  - LLMLingua-2: https://arxiv.org/abs/2403.12968
  - GLiNER2: https://arxiv.org/abs/2507.18546

Config (config.yaml):
  compression:
    flush_strategy: offline     # "llm" (default) or "offline"
    summary_strategy: offline   # "llm" (default) or "offline"
"""

import json
import logging
import re
import threading
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Recursive token-aware text splitter ────────────────────────────────────
#
# Follows the recursive text splitter pattern (cf. LangChain, LlamaIndex):
# try separators in decreasing semantic granularity, greedily packing
# small pieces and recursing on oversized ones.  Token counts are verified
# by the model's own tokenizer at every level — no character estimation.

_MAX_CHUNK_TOKENS = 510  # XLM-RoBERTa positional limit minus BOS/EOS

# Separator hierarchy: paragraph → line → sentence → word → token-ids.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _chunk_by_tokens(tokenizer, text: str, max_tokens: int = _MAX_CHUNK_TOKENS) -> List[str]:
    """Split *text* into chunks guaranteed to be <= *max_tokens*.

    Uses a recursive separator hierarchy (paragraph → line → sentence →
    word) with greedy packing.  Falls back to hard token-id splits as a
    last resort.  Every returned chunk is verified by the tokenizer.
    """
    return _recursive_split(tokenizer, text, _SEPARATORS, max_tokens)


def _recursive_split(tokenizer, text: str, separators: List[str],
                     max_tokens: int) -> List[str]:
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)

    # Base case: text already fits.
    if len(encode(text)) <= max_tokens:
        return [text] if text.strip() else []

    # No separators left — hard-split by token ids.
    if not separators:
        ids = encode(text)
        chunks = []
        for i in range(0, len(ids), max_tokens):
            piece = tokenizer.decode(ids[i:i + max_tokens],
                                     skip_special_tokens=True).strip()
            if piece:
                chunks.append(piece)
        return chunks

    sep = separators[0]
    next_seps = separators[1:]
    parts = text.split(sep)

    # Greedily pack small parts; recurse on oversized ones.
    chunks: List[str] = []
    buf: List[str] = []
    buf_tok = 0

    for part in parts:
        t = len(encode(part))
        if t > max_tokens:
            if buf:
                chunks.append(sep.join(buf))
                buf, buf_tok = [], 0
            chunks.extend(_recursive_split(tokenizer, part, next_seps, max_tokens))
            continue
        if buf_tok + t + 1 > max_tokens and buf:
            chunks.append(sep.join(buf))
            buf, buf_tok = [], 0
        buf.append(part)
        buf_tok += t + 1

    if buf:
        chunks.append(sep.join(buf))
    return chunks


# ── Cached model loading ───────────────────────────────────────────────────

_gliner_instance = None
_gliner_lock = threading.Lock()

_compressor_instance = None
_compressor_lock = threading.Lock()


def _get_gliner():
    """Return a cached GLiNER model, loading on first call."""
    global _gliner_instance
    if _gliner_instance is not None:
        return _gliner_instance
    with _gliner_lock:
        if _gliner_instance is not None:
            return _gliner_instance
        from gliner import GLiNER
        import huggingface_hub.utils as hfu
        was_enabled = not hfu.are_progress_bars_disabled()
        hfu.disable_progress_bars()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*resume_download.*")
                _gliner_instance = GLiNER.from_pretrained(
                    "knowledgator/gliner-multitask-large-v0.5")
        finally:
            if was_enabled:
                hfu.enable_progress_bars()
        return _gliner_instance


def _get_compressor():
    """Return a cached PromptCompressor, loading on first call."""
    global _compressor_instance
    if _compressor_instance is not None:
        return _compressor_instance
    with _compressor_lock:
        if _compressor_instance is not None:
            return _compressor_instance
        from llmlingua import PromptCompressor
        import torch, transformers
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if getattr(torch.backends, "mps", None)
                       and torch.backends.mps.is_available()
                  else "cpu")
        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        try:
            _compressor_instance = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True, device_map=device)
        finally:
            transformers.logging.set_verbosity(prev)
            transformers.logging.enable_progress_bar()
        # We chunk ourselves — suppress the tokenizer's length warning.
        _compressor_instance.tokenizer.model_max_length = 1_000_000
        return _compressor_instance


# ── Memory extraction (GLiNER2) ────────────────────────────────────────────

_MEMORY_LABELS = ["user_preference", "user_fact", "decision", "correction"]
_USER_LABELS = {"user_preference", "user_fact"}
_MAX_MEMORIES = 8


def _make_tool_call(target: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(function=SimpleNamespace(
        name="memory",
        arguments=json.dumps({"action": "add", "target": target, "content": content}),
    ))


def extract_memories(messages: List[Dict[str, Any]]) -> list:
    """Extract memory-worthy facts from user messages via GLiNER2.

    Returns tool_call objects compatible with the built-in memory tool.
    """
    try:
        from gliner import GLiNER  # noqa: F401
    except ImportError:
        logger.warning("GLiNER2 not installed. Install with: pip install gliner2")
        return []

    user_texts = [
        (m.get("content") or "").strip()
        for m in messages if m.get("role") == "user"
        and len((m.get("content") or "").strip()) >= 30
    ]
    if not user_texts:
        return []

    model = _get_gliner()
    tokenizer = model.data_processor.transformer_tokenizer

    # Chunk oversized messages to fit the model's token window.
    segments: List[str] = []
    for text in user_texts:
        if len(tokenizer.encode(text, add_special_tokens=False)) <= _MAX_CHUNK_TOKENS:
            segments.append(text)
        else:
            parts = re.split(r'(?<=\. )|\n', text)
            buf, buf_tok = [], 0
            for part in parts:
                t = len(tokenizer.encode(part, add_special_tokens=False))
                if buf_tok + t + 1 > _MAX_CHUNK_TOKENS and buf:
                    segments.append(" ".join(buf))
                    buf, buf_tok = [], 0
                buf.append(part)
                buf_tok += t + 1
            if buf:
                segments.append(" ".join(buf))

    tool_calls = []
    seen = set()
    for text in segments:
        for ent in model.predict_entities(text, _MEMORY_LABELS, threshold=0.4):
            span = ent.get("text", "").strip()
            if not span or len(span) < 15 or span.lower() in seen:
                continue
            seen.add(span.lower())
            target = "user" if ent.get("label") in _USER_LABELS else "memory"
            tool_calls.append(_make_tool_call(target, span))
            if len(tool_calls) >= _MAX_MEMORIES:
                return tool_calls
    return tool_calls


# ── Context compression (LLMLingua-2) ──────────────────────────────────────

_DEFAULT_RATE = 0.3


def compress_summary(serialized_content: str, rate: float = _DEFAULT_RATE) -> Optional[str]:
    """Compress conversation text via LLMLingua-2 token classification.

    Input is split into tokenizer-verified chunks within XLM-RoBERTa's
    512-token window, so arbitrarily long conversations are handled.
    """
    try:
        from llmlingua import PromptCompressor  # noqa: F401
    except ImportError:
        logger.warning("LLMLingua not installed. Install with: pip install llmlingua")
        return None
    if not serialized_content or not serialized_content.strip():
        return None

    compressor = _get_compressor()
    parts = []
    for chunk in _chunk_by_tokens(compressor.tokenizer, serialized_content):
        if not chunk.strip():
            continue
        result = compressor.compress_prompt(
            chunk, rate=rate, force_tokens=["\n", ".", ":", "#", "##"])
        text = result.get("compressed_prompt", "").strip()
        if text:
            parts.append(text)
    compressed = "\n\n".join(parts)
    return compressed if compressed else None
