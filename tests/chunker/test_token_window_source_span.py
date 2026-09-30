"""Regression tests for the O(N) delta-decode in ``_token_window_source_span``.

The fixed-token chunker maps each decoded token window back to its exact source
span. The offset of a window's start used to be recomputed with a full
``decode(tokens[:start_token])`` prefix decode on every window — O(N) per window,
O(N²) over a document. It now decodes only the delta since the previous *verified*
anchor (O(N) total). These tests pin three properties:

1. **Equivalence** — the delta path yields byte-identical spans to a reference
   full-prefix implementation, window for window (incl. a non-1:1 tokenizer and a
   real tiktoken BPE tokenizer).
2. **Exactness** — every emitted span is a verbatim slice of the source.
3. **Linear decode budget** — total tokens handed to ``decode`` scales ~O(N), not
   O(N²); the pre-optimization prefix decode would blow well past the bound.
"""

from __future__ import annotations

import pytest

from lightrag.chunker import chunking_by_fixed_token
from lightrag.chunker.token_size import (
    _replacement_edge_match,
    _source_span,
    _token_window_source_span,
    chunking_by_token_size,
)
from lightrag.utils import Tokenizer, TokenizerInterface


class _CharTokenizer(TokenizerInterface):
    """1:1 char-per-token; ``decode(encode(x)) == x`` so windows are verbatim."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class _MultiTokenTokenizer(TokenizerInterface):
    """Non-uniform char→token ratio: uppercase = 2 tokens, ``.?!`` = 3, else 1.

    Exercises the offset arithmetic where token count != char count, so a bug that
    confuses the two surfaces immediately.
    """

    def encode(self, content: str) -> list[int]:
        tokens: list[int] = []
        for ch in content:
            if ch.isupper():
                tokens.extend([ord(ch), ord(ch) + 1000])
            elif ch in (".", "?", "!"):
                tokens.extend([ord(ch), ord(ch) + 2000, ord(ch) + 3000])
            else:
                tokens.append(ord(ch))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        result: list[str] = []
        i = 0
        while i < len(tokens):
            base = tokens[i]
            if (
                i + 2 < len(tokens)
                and tokens[i + 1] == base + 2000
                and tokens[i + 2] == base + 3000
            ):
                result.append(chr(base))
                i += 3
            elif i + 1 < len(tokens) and tokens[i + 1] == base + 1000:
                result.append(chr(base))
                i += 2
            else:
                result.append(chr(base))
                i += 1
        return "".join(result)


class _CountingTokenizer(TokenizerInterface):
    """Wraps a char tokenizer and tallies every token handed to ``decode``."""

    def __init__(self) -> None:
        self.decoded_tokens = 0

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        self.decoded_tokens += len(tokens)
        return "".join(chr(t) for t in tokens)


def _ref_full_prefix_span(
    tokenizer: Tokenizer,
    content: str,
    tokens: list[int],
    start_token: int,
    end_token: int,
) -> dict[str, int] | None:
    """The pre-optimization implementation: full ``decode(tokens[:start_token])``."""
    window = tokenizer.decode(tokens[start_token:end_token])
    start = len(tokenizer.decode(tokens[:start_token]))
    end = start + len(window)
    if content[start:end] != window:
        found = content.find(
            window, max(0, start - 32), min(len(content), end + 32 + len(window))
        )
        if found < 0:
            return None
        start, end = found, found + len(window)
    return _source_span(content, start, end)


def _windows(n_tokens: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    step = chunk_size - overlap
    return [
        (start, min(start + chunk_size, n_tokens)) for start in range(0, n_tokens, step)
    ]


@pytest.mark.offline
@pytest.mark.parametrize(
    "content",
    [
        " ".join(f"word{i:03d}" for i in range(400)),
        "Alpha. BETA gamma? Delta! " * 120,
        "Repeated phrase. " * 200,  # heavily repeated — exact offsets must not drift
    ],
)
def test_delta_decode_matches_full_prefix_reference(content: str) -> None:
    for impl in (_CharTokenizer(), _MultiTokenTokenizer()):
        tok = Tokenizer(model_name="t", tokenizer=impl)
        tokens = tok.encode(content)
        chunk_size, overlap = 60, 12
        anchor = (0, 0)
        for start_token, end_token in _windows(len(tokens), chunk_size, overlap):
            got, anchor = _token_window_source_span(
                tok, content, tokens, start_token, end_token, anchor=anchor
            )
            ref = _ref_full_prefix_span(tok, content, tokens, start_token, end_token)
            assert got == ref


@pytest.mark.offline
def test_delta_decode_matches_full_prefix_with_tiktoken() -> None:
    pytest.importorskip("tiktoken")
    from lightrag.utils import TiktokenTokenizer

    tok = TiktokenTokenizer()
    content = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
    ) * 40
    tokens = tok.encode(content)
    anchor = (0, 0)
    for start_token, end_token in _windows(len(tokens), 24, 6):
        got, anchor = _token_window_source_span(
            tok, content, tokens, start_token, end_token, anchor=anchor
        )
        ref = _ref_full_prefix_span(tok, content, tokens, start_token, end_token)
        assert got == ref


@pytest.mark.offline
def test_fixed_token_spans_are_exact_on_long_doc() -> None:
    content = " ".join(f"token{i:04d}" for i in range(1500))
    tok = Tokenizer(model_name="char", tokenizer=_CharTokenizer())

    chunks = chunking_by_fixed_token(
        tok,
        content,
        chunk_token_size=120,
        chunk_overlap_token_size=20,
        _emit_source_span=True,
    )

    assert len(chunks) > 5
    for chunk in chunks:
        span = chunk["_source_span"]
        assert content[span["start"] : span["end"]] == chunk["content"]


@pytest.mark.offline
def test_decode_budget_is_linear_not_quadratic() -> None:
    # char tokenizer => len(tokens) == len(content). With the old full-prefix
    # decode the helper alone would decode ~sum(starts) == N^2/(2*step) tokens;
    # the delta path decodes ~one step per window plus each window once. Assert
    # the total stays under a small linear multiple of N, a bound the quadratic
    # implementation cannot meet.
    content = "x" * 8000
    counting = _CountingTokenizer()
    tok = Tokenizer(model_name="counting", tokenizer=counting)
    n = len(tok.encode(content))  # 8000

    chunking_by_fixed_token(
        tok,
        content,
        chunk_token_size=200,
        chunk_overlap_token_size=20,
        _emit_source_span=True,
    )

    # Empirically ~3.2*N for the delta path; the old prefix decode is ~24*N here.
    assert counting.decoded_tokens <= 6 * n


def _tiktoken():
    """Real tiktoken, the tokenizer the report was produced against.

    A synthetic byte-per-token tokenizer was tried first and is too
    pathological: with a 140-byte window the decoded text is ~105 characters,
    so the +/-32 fallback around the predicted start is exhausted by ordinary
    drift and unrelated windows lose their span. Real BPE averages ~4 bytes
    per token, and the anchoring was built for that.
    """
    from lightrag.utils import TiktokenTokenizer

    return Tokenizer(model_name="tiktoken", tokenizer=TiktokenTokenizer())


def _multibyte_document() -> str:
    para = (
        "Release notes: 🎉 chunker ✅ shipped. "
        "Families 👨👩👧👦 and flags 🇨🇳 🇯🇵 appear in user text. "
    )
    return para * 24


@pytest.mark.offline
def test_window_splitting_a_character_keeps_its_span():
    """A window whose boundary cuts a multi-byte character still resolves.

    The decoded window differs from the source slice only by the U+FFFD that
    replaced the split character, so trimming those before the probe and
    widening the span back over them recovers the exact position.
    """
    content = _multibyte_document()
    tok = _tiktoken()

    chunks = chunking_by_token_size(
        tok,
        content,
        chunk_token_size=140,
        chunk_overlap_token_size=20,
        _emit_source_span=True,
    )

    assert len(chunks) > 1
    for chunk in chunks:
        span = chunk.get("_source_span")
        assert span is not None, (
            f"chunk {chunk['chunk_order_index']} lost its span; "
            f"content carries {chunk['content'].count(chr(0xFFFD))} replacement char(s)"
        )
        # The span covers the real characters; the chunk's own text carries
        # U+FFFD where a boundary split one, so byte equality is the wrong
        # assertion -- the replacement-edge rule is.
        assert (
            _replacement_edge_match(
                content[span["start"] : span["end"]], chunk["content"]
            )
            or content[span["start"] : span["end"]] == chunk["content"].strip()
        )


@pytest.mark.offline
def test_no_gap_is_left_between_consecutive_spans_on_multibyte_text():
    """The regression that matters: consecutive windows must not leave a hole.

    Losing a span does not only cost that one chunk -- the surviving spans do
    not reach across the gap, so a run of consecutive windows left a
    contiguous stretch of the document covered by no chunk at all.
    """
    content = _multibyte_document()
    tok = _tiktoken()

    chunks = chunking_by_token_size(
        tok,
        content,
        chunk_token_size=140,
        chunk_overlap_token_size=20,
        _emit_source_span=True,
    )

    covered = bytearray(len(content))
    for chunk in chunks:
        span = chunk.get("_source_span")
        assert span is not None, f"chunk {chunk['chunk_order_index']} lost its span"
        for i in range(span["start"], span["end"]):
            covered[i] = 1

    gaps, run_start = [], None
    for i, hit in enumerate(covered):
        if not hit and run_start is None:
            run_start = i
        elif hit and run_start is not None:
            gaps.append((run_start, i))
            run_start = None
    if run_start is not None:
        gaps.append((run_start, len(covered)))

    # Only trailing/leading whitespace may sit outside a span; spans are trimmed.
    gaps = [(s, e) for s, e in gaps if content[s:e].strip() != ""]
    assert not gaps, f"non-whitespace text covered by no chunk span: {gaps}"


@pytest.mark.offline
def test_clean_window_keeps_the_span_the_find_fallback_located():
    """The +/-32 find fallback must keep its verdict for a clean window.

    Regression guard. The replacement-edge predicate was at first applied to
    *every* window, including ones with no U+FFFD. Such a window that missed
    its predicted offset but was then located by the fallback was rejected
    anyway, and because its body carries no replacement character
    `_is_unlocatable` said the chunk was fine, so backfill raised
    ChunkBlockMatchError and failed a document that previously degraded to a
    missing sidecar.

    Driving the helper directly keeps the test deterministic: it does not
    depend on finding a document that happens to split a character, which is
    why a whole-document fixture could not reproduce this.
    """
    content = "".join(chr(ord("a") + (i % 26)) for i in range(200))
    tok = Tokenizer(model_name="char", tokenizer=_CharTokenizer())
    start_token, end_token = 20, 60
    # A deliberately wrong anchor: 5 characters too far along, which is what a
    # drifted anchor looks like in practice.
    span, _new_anchor = _token_window_source_span(
        tok,
        content,
        tok.encode(content),
        start_token,
        end_token,
        anchor=(0, 5),
    )
    assert span is not None, (
        "a clean window located by the +/-32 fallback was discarded; that turns "
        "'found it a few characters away' into 'give up' and fails the document"
    )
    assert content[span["start"] : span["end"]] == content[start_token:end_token]
