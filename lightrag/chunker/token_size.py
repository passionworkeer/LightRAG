"""Fixed-size token-window chunking — the LightRAG default strategy.

Chunks the input text into windows of at most ``chunk_token_size`` tokens
with ``chunk_overlap_token_size`` of overlap between adjacent windows.
When ``split_by_character`` is supplied, the splitter first segments on
that delimiter and then either tokenizes each segment as-is
(``split_by_character_only=True``) or further sub-splits any segment
that exceeds the token cap.

Two entry points are exported:

  - :func:`chunking_by_token_size` — the **legacy 6-arg signature**
    used as the default value for :attr:`lightrag.LightRAG.chunking_func`.
    Kept for backward compatibility so externally-supplied chunking
    functions can continue to drop in unchanged.

  - :func:`chunking_by_fixed_token` — the same algorithm exposed under
    the **new file-chunker contract** (standard prefix
    ``(tokenizer, content, chunk_token_size)`` plus keyword-only
    knobs). Used by the file-based chunking dispatcher in
    ``process_single_document`` for ``doc_process_opts.chunking == "F"``.
"""

from __future__ import annotations

from typing import Any

from lightrag.exceptions import ChunkTokenLimitExceededError
from lightrag.utils import Tokenizer, logger


def _trimmed_span(content: str, start: int, end: int) -> tuple[int, int]:
    """Return the source span after applying the chunker's ``.strip()``."""
    start = max(0, min(start, len(content)))
    end = max(start, min(end, len(content)))
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    return start, end


def _source_span(content: str, start: int, end: int) -> dict[str, int] | None:
    start, end = _trimmed_span(content, start, end)
    if start >= end:
        return None
    return {"start": start, "end": end}


_REPLACEMENT_CHAR = "�"


def _replacement_edge_match(source_text: str, body: str) -> bool:
    """True when two texts differ only by replacement characters at the edges.

    A token window whose boundary split a multi-byte character decodes that
    character as U+FFFD, so the chunk's content and its source span differ at
    the edges only. Each side accounts for exactly one source character --
    a window boundary cannot split more than that -- but the *number* of
    U+FFFD is meaningless: a 4-byte character with 1, 2 or 3 of its bytes
    inside the window decodes to 1, 2 or 3 of them. So the two sides are
    aligned by "one character per side that has a replacement", not by
    counting them.

    An interior U+FFFD is never accepted: the replacement character only
    appears where a window begins or ends, so one in the middle means the
    chunk really did lose bytes.

    Shared with ``lightrag.sidecar.backfill`` so the chunker and the
    validator cannot drift apart -- they are two halves of one rule.
    """
    if _REPLACEMENT_CHAR not in body:
        return False
    core = body.strip(_REPLACEMENT_CHAR)
    if not core:
        return False
    head = 1 if body.startswith(_REPLACEMENT_CHAR) else 0
    tail = 1 if body.endswith(_REPLACEMENT_CHAR) else 0
    if len(source_text) != len(core) + head + tail:
        return False
    return (
        source_text[head : len(source_text) - tail if tail else len(source_text)]
        == core
    )


def _token_window_source_span(
    tokenizer: Tokenizer,
    content: str,
    tokens: list[int],
    start_token: int,
    end_token: int,
    *,
    anchor: tuple[int, int],
) -> tuple[dict[str, int] | None, tuple[int, int]]:
    """Map a decoded token window back to its exact source span.

    ``anchor`` is the previous window's *verified* ``(start_token, start_char)``.
    Window starts are monotonically increasing, so instead of re-decoding the whole
    ``tokens[:start_token]`` prefix (O(N) per window → O(N²) overall) we decode only
    the delta ``tokens[anchor_token:start_token]`` (≈ one chunking step) to predict
    the start char. The predicted offset is then verified against ``content`` exactly
    as a full prefix decode would be: byte-level BPE decode is non-concatenative at a
    multi-byte UTF-8 boundary, so a delta can be off by the few chars of one split
    char — the ±32 ``find`` fallback corrects that, and re-anchoring on the verified
    position each call keeps the error from accumulating. Net cost is O(N) total
    while the located span stays byte-exact.

    Returns ``(span, new_anchor)``. A window whose boundary split a multi-byte
    character still resolves, because the replacement characters are trimmed
    before the probe and the span is widened back over them; a window that is
    nothing *but* replacement characters stays unlocatable and the anchor is
    returned unchanged so the next window still predicts from the last
    verified position.
    """
    anchor_token, anchor_char = anchor
    window = tokenizer.decode(tokens[start_token:end_token])
    # A multi-byte char split at either window boundary decodes to U+FFFD, so
    # the decoded window differs from the source slice only at its edges. How
    # *many* U+FFFD one split character produces depends on how many of its
    # bytes landed inside the window (a 4-byte char can leave 1, 2 or 3), so the
    # replacement characters cannot be counted to size the widening -- only
    # their presence matters, because a window boundary splits at most one
    # character per side. Probe with them trimmed and widen by one character
    # per side that had one.
    core = window.strip(_REPLACEMENT_CHAR)
    lead = 1 if window.startswith(_REPLACEMENT_CHAR) else 0
    trail = 1 if window.endswith(_REPLACEMENT_CHAR) else 0
    if start_token >= anchor_token:
        start = anchor_char + len(tokenizer.decode(tokens[anchor_token:start_token]))
    else:  # non-monotonic caller (not expected) — fall back to a full prefix decode
        start = len(tokenizer.decode(tokens[:start_token]))
    end = start + len(window)
    if content[start:end] != window:
        if not core:
            return None, anchor
        lo = max(0, start - 32)
        hi = min(len(content), end + 32 + len(core))
        # find() returns only the first hit, and a window carrying replacement
        # characters searches for a shorter needle -- a repeated phrase or a
        # common token run is therefore more likely to produce a near-miss here
        # than for a clean window. Keep looking instead of giving up on the
        # first candidate that does not line up.
        search_from = lo
        while True:
            found = content.find(core, search_from, hi)
            if found < 0:
                return None, anchor
            start = found - lead
            end = found + len(core) + trail
            if (
                start >= 0
                and end <= len(content)
                and (
                    core == window
                    or _replacement_edge_match(content[start:end], window)
                )
            ):
                break
            search_from = found + 1
    return _source_span(content, start, end), (start_token, start)


def _make_chunk(
    *,
    content: str,
    tokens: int,
    order: int,
    source_span: dict[str, int] | None,
    emit_source_span: bool,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "tokens": tokens,
        "content": content.strip(),
        "chunk_order_index": order,
    }
    if emit_source_span and source_span is not None:
        item["_source_span"] = source_span
    return item


def _window_step(chunk_token_size: int, chunk_overlap_token_size: int) -> int:
    """Token-window stride for the sliding-window chunk loops.

    When overlap >= size the stride is <= 0, which makes ``range()`` yield an
    empty sequence (dropping the whole segment silently) or raise the opaque
    ``range() arg 3 must not be zero``. Fail closed with the same invariant the
    API-boundary validators enforce.

    Negative overlap is rejected up front in ``chunking_by_token_size()``,
    before this function is ever reached -- see the check there.
    """
    if chunk_overlap_token_size >= chunk_token_size:
        raise ValueError(
            f"chunk_overlap_token_size ({chunk_overlap_token_size}) must be < "
            f"chunk_token_size ({chunk_token_size})"
        )
    return chunk_token_size - chunk_overlap_token_size


def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
    *,
    _emit_source_span: bool = False,
) -> list[dict[str, Any]]:
    """Legacy 6-arg fixed-token chunker (default for ``LightRAG.chunking_func``).

    Signature is preserved for backward compatibility with externally
    supplied ``chunking_func`` implementations. New file-based chunking
    dispatch uses :func:`chunking_by_fixed_token` instead.
    """
    # Checked up front, before any branching on split_by_character, so a
    # negative chunk_overlap_token_size is always rejected -- including on
    # the split_by_character path where every segment stays under
    # chunk_token_size and _window_step()'s own (upper-bound only) check
    # would otherwise never be reached. The upper-bound check stays lazy,
    # inside _window_step(), since it's only meaningful once windowing is
    # actually about to happen.
    if chunk_overlap_token_size < 0:
        raise ValueError(
            f"chunk_overlap_token_size must be non-negative, got "
            f"{chunk_overlap_token_size}"
        )
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        raw_spans: list[tuple[int, int]] = []
        cursor = 0
        for raw_chunk in raw_chunks:
            start = cursor
            end = start + len(raw_chunk)
            raw_spans.append((start, end))
            cursor = end + len(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk, (chunk_start, chunk_end) in zip(raw_chunks, raw_spans):
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    logger.warning(
                        "Chunk split_by_character exceeds token limit: len=%d limit=%d",
                        len(_tokens),
                        chunk_token_size,
                    )
                    raise ChunkTokenLimitExceededError(
                        chunk_tokens=len(_tokens),
                        chunk_token_limit=chunk_token_size,
                        chunk_preview=chunk[:120],
                    )
                span = (
                    _source_span(content, chunk_start, chunk_end)
                    if _emit_source_span
                    else None
                )
                new_chunks.append((len(_tokens), chunk, span))
        else:
            for chunk, (chunk_start, chunk_end) in zip(raw_chunks, raw_spans):
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    # Anchor is chunk-relative (offsets are shifted by chunk_start
                    # below), so it resets per split-by-character segment.
                    anchor = (0, 0)
                    for start in range(
                        0,
                        len(_tokens),
                        _window_step(chunk_token_size, chunk_overlap_token_size),
                    ):
                        end_token = min(start + chunk_token_size, len(_tokens))
                        chunk_content = tokenizer.decode(_tokens[start:end_token])
                        span = None
                        if _emit_source_span:
                            span, anchor = _token_window_source_span(
                                tokenizer,
                                chunk,
                                _tokens,
                                start,
                                end_token,
                                anchor=anchor,
                            )
                        if span is not None:
                            span = {
                                "start": chunk_start + span["start"],
                                "end": chunk_start + span["end"],
                            }
                        new_chunks.append(
                            (
                                min(chunk_token_size, len(_tokens) - start),
                                chunk_content,
                                span,
                            )
                        )
                else:
                    span = (
                        _source_span(content, chunk_start, chunk_end)
                        if _emit_source_span
                        else None
                    )
                    new_chunks.append((len(_tokens), chunk, span))
        for index, (_len, chunk, span) in enumerate(new_chunks):
            results.append(
                _make_chunk(
                    content=chunk,
                    tokens=_len,
                    order=index,
                    source_span=span,
                    emit_source_span=_emit_source_span,
                )
            )
    else:
        # Only the fixed-window path needs the whole-document token list;
        # the split_by_character path re-encodes each segment anyway (BPE
        # boundaries differ on substrings), so the full encode is deferred
        # here rather than paid on every call.
        tokens = tokenizer.encode(content)
        anchor = (0, 0)
        for index, start in enumerate(
            range(
                0,
                len(tokens),
                _window_step(chunk_token_size, chunk_overlap_token_size),
            )
        ):
            end = min(start + chunk_token_size, len(tokens))
            chunk_content = tokenizer.decode(tokens[start:end])
            span = None
            if _emit_source_span:
                span, anchor = _token_window_source_span(
                    tokenizer, content, tokens, start, end, anchor=anchor
                )
            results.append(
                _make_chunk(
                    content=chunk_content,
                    tokens=min(chunk_token_size, len(tokens) - start),
                    order=index,
                    source_span=span,
                    emit_source_span=_emit_source_span,
                )
            )
    return results


def chunking_by_fixed_token(
    tokenizer: Tokenizer,
    content: str,
    chunk_token_size: int = 1200,
    *,
    chunk_overlap_token_size: int = 100,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    _emit_source_span: bool = False,
) -> list[dict[str, Any]]:
    """Fixed-token chunker — file-chunker contract for the ``"F"`` strategy.

    Implements the same fixed-window algorithm as
    :func:`chunking_by_token_size`, exposed under the standard
    file-chunker signature ``(tokenizer, content, chunk_token_size, *,
    <strategy kwargs>)`` so the file-based chunking dispatcher in
    ``process_single_document`` can call every strategy uniformly.
    """
    return chunking_by_token_size(
        tokenizer,
        content,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        chunk_overlap_token_size=chunk_overlap_token_size,
        chunk_token_size=chunk_token_size,
        _emit_source_span=_emit_source_span,
    )
