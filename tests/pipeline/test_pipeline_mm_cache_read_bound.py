"""Regression tests for the multimodal cache-read fan-out in analyze_multimodal.

Every sidecar item is scheduled as its own task, so the number of
concurrent ``handle_cache`` calls tracks the number of items in the
sidecar -- not the configured VLM/EXTRACT concurrency.  On a Redis KV
store that turns one document with N drawings into N simultaneous pool
checkouts, which is how #4243 exhausted ``REDIS_MAX_CONNECTIONS`` and
masked the real error behind a "cancelled" sibling.

These tests pin the bound itself (peak in-flight reads) rather than a
storage error message, so they hold for every KV backend.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


from tests.pipeline.test_pipeline_analyze_multimodal import (  # noqa: F401
    PNG_BYTES,
    ROLES,
    EmbeddingFunc,
    LightRAG,
    RoleLLMConfig,
    Tokenizer,
    _build_rag,
    _mock_embedding,
    _SimpleTokenizerImpl,
)

pytestmark = pytest.mark.offline

# An independent sanity ceiling, NOT the implementation's value (5). Two
# reasons. First, this file must also run against an unfixed tree: importing
# a constant that only exists after the fix turns a red assertion into an
# ImportError at collection, which judges nothing. Second, keeping the
# number independent means the assertion states a property ("the peak stays
# small") rather than restating whatever the code happens to do.
#
# Baseline peaks at the item count -- 24 in test 1, 32 in test 2 -- so an
# unbounded tree is red by a wide margin, not by a rounding error.
_CACHE_READ_CEILING = 8


class _WriteProbe:
    """Counts peak concurrent in-flight cache WRITES.

    ``save_to_cache`` is two round-trips per call, so the write fan-out is the
    larger half of the storage pressure even at the same item count.
    """

    def __init__(self, hold: float = 0.02) -> None:
        self.in_flight = 0
        self.peak = 0
        self.total = 0
        self._hold = hold

    async def write(self, *args, **kwargs) -> None:
        self.in_flight += 1
        self.total += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self._hold)
        finally:
            self.in_flight -= 1


class _ConcurrencyProbe:
    """Counts peak concurrent in-flight cache reads.

    ``hold`` keeps each read open long enough that an unbounded scheduler
    can pile up, which a fast backend would otherwise hide by completing
    one read before the next task starts.
    """

    def __init__(self, hold: float = 0.02) -> None:
        self.in_flight = 0
        self.peak = 0
        self.total = 0
        self._hold = hold

    async def read(self) -> tuple[None, int]:
        self.in_flight += 1
        self.total += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self._hold)
            return None, 0
        finally:
            self.in_flight -= 1


def _write_table_sidecar(tmp_path: Path, *, items: int) -> tuple[str, dict, Path]:
    """Build a blocks file plus a tables sidecar holding ``items`` entries."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir(exist_ok=True)

    blocks_path = parsed_dir / "doc.blocks.jsonl"
    blocks_path.write_text(
        json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n",
        encoding="utf-8",
    )

    # ``analyze_multimodal`` validates each table item with ``item.get("format")``
    # and accepts only "html" or "json" -- anything else raises before the cache
    # read this file is meant to measure. The key must be exactly "format".
    tables = {
        f"tbl-{i:03d}": {
            "caption": f"Table {i}",
            "content": f"<table><tr><td>{i}</td><td>{i}</td></tr></table>",
            "format": "html",
        }
        for i in range(items)
    }
    sidecar_path = parsed_dir / "doc.tables.json"
    sidecar_path.write_text(json.dumps({"tables": tables}), encoding="utf-8")

    return "doc-1", {"blocks_path": str(blocks_path)}, sidecar_path


def _build_extract_rag(tmp_path: Path, extract_func) -> LightRAG:
    role_configs = {spec.name: RoleLLMConfig() for spec in ROLES}
    for spec in ROLES:
        if spec.name == "extract":
            role_configs[spec.name] = RoleLLMConfig(func=extract_func)
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"mm-cache-probe-{tmp_path.name}",
        llm_model_func=extract_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=1024,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        role_llm_configs=role_configs,
    )


@pytest.mark.asyncio
async def test_cache_reads_stay_bounded_regardless_of_item_count(tmp_path, monkeypatch):
    """Peak concurrent cache reads must not grow with the sidecar size.

    Baseline behaviour schedules one task per item, so ``peak`` reaches
    the item count.  The fix bounds the reads, so peak stays at the
    configured ceiling no matter how big the document is.
    """
    import lightrag.pipeline as pipeline_mod

    item_count = 24
    assert item_count > _CACHE_READ_CEILING * 2, "item count must exceed the bound"

    probe = _ConcurrencyProbe()

    async def extract_func(prompt, **kwargs):
        return json.dumps({"name": "n", "description": "d"})

    doc_id, parsed_data, sidecar_path = _write_table_sidecar(tmp_path, items=item_count)

    rag = _build_extract_rag(tmp_path, extract_func)
    await rag.initialize_storages()
    try:
        write_probe = _WriteProbe()
        monkeypatch.setattr(pipeline_mod, "handle_cache", lambda *a, **k: probe.read())
        monkeypatch.setattr(
            pipeline_mod, "save_to_cache", lambda *a, **k: write_probe.write()
        )
        # Keep every item on the fresh-LLM path so nothing is served from
        # the on-disk cache: we are counting reads, not hits.
        await rag.analyze_multimodal(
            doc_id=doc_id,
            file_path="demo.pdf",
            parsed_data=parsed_data,
            process_options="t",
        )
    finally:
        await rag.finalize_storages()

    assert probe.total == item_count, "every item must still be analyzed"
    assert probe.peak <= _CACHE_READ_CEILING, (
        f"peak concurrent cache reads was {probe.peak}, bound is "
        f"{_CACHE_READ_CEILING} -- cache reads are not bounded independently "
        f"of the sidecar size"
    )
    # The write side needs its own bound: save_to_cache is two round-trips per
    # item, so gating reads alone leaves the connection pressure that #4243
    # reports largely in place.
    assert write_probe.total == item_count, (
        f"expected one cache write per item, saw {write_probe.total}"
    )
    assert write_probe.peak <= _CACHE_READ_CEILING, (
        f"peak concurrent cache writes was {write_probe.peak}, bound is "
        f"{_CACHE_READ_CEILING} -- cache writes are not bounded independently "
        f"of the sidecar size"
    )


@pytest.mark.asyncio
async def test_cache_writes_stay_bounded_regardless_of_item_count(
    tmp_path, monkeypatch
):
    """Peak concurrent cache WRITES must not grow with the sidecar size.

    Deliberately not folded into the read test. With reads gated, the pipeline's
    rhythm keeps writes under the ceiling anyway, so a write assertion there
    passes even when the write gate is absent -- the read gate masks the write
    gate. Here reads return immediately and writes hold, so the writes are the
    bottleneck and nothing else can hold the peak down.

    ``save_to_cache`` costs two Redis round-trips per call, so the write side is
    the larger half of the storage pressure in the report #4243 is about.
    """
    import json

    import lightrag.pipeline as pipeline_mod

    item_count = 24
    assert item_count > _CACHE_READ_CEILING * 2, "item count must exceed the bound"

    read_probe = _ConcurrencyProbe(hold=0.0)
    write_probe = _WriteProbe(hold=0.02)

    async def extract_func(prompt, **kwargs):
        return json.dumps({"name": "n", "description": "d"})

    doc_id, parsed_data, _ = _write_table_sidecar(tmp_path, items=item_count)

    rag = _build_extract_rag(tmp_path, extract_func)
    await rag.initialize_storages()
    try:
        monkeypatch.setattr(
            pipeline_mod, "handle_cache", lambda *a, **k: read_probe.read()
        )
        monkeypatch.setattr(
            pipeline_mod, "save_to_cache", lambda *a, **k: write_probe.write()
        )
        await rag.analyze_multimodal(
            doc_id=doc_id,
            file_path="demo.pdf",
            parsed_data=parsed_data,
            process_options="t",
        )
    finally:
        await rag.finalize_storages()

    assert read_probe.total == item_count, (
        f"expected one cache read per item, saw {read_probe.total}"
    )
    assert write_probe.total == item_count, (
        f"expected one cache write per item, saw {write_probe.total}"
    )
    assert write_probe.peak <= _CACHE_READ_CEILING, (
        f"peak concurrent cache writes was {write_probe.peak}, bound is "
        f"{_CACHE_READ_CEILING} -- cache writes are not bounded independently "
        f"of the sidecar size"
    )


@pytest.mark.asyncio
async def test_peak_cache_reads_do_not_scale_with_document_size(tmp_path, monkeypatch):
    """Doubling the sidecar must not double peak concurrent cache reads.

    A bound that merely grows more slowly would still pass the single
    -size assertion above; this pins the shape of the fix.
    """
    import lightrag.pipeline as pipeline_mod

    async def extract_func(prompt, **kwargs):
        return json.dumps({"name": "n", "description": "d"})

    peaks: dict[int, int] = {}
    for item_count in (8, 32):
        case_dir = tmp_path / f"n{item_count}"
        case_dir.mkdir()
        probe = _ConcurrencyProbe()
        doc_id, parsed_data, _ = _write_table_sidecar(case_dir, items=item_count)
        rag = _build_extract_rag(case_dir, extract_func)
        await rag.initialize_storages()
        try:
            monkeypatch.setattr(
                pipeline_mod, "handle_cache", lambda *a, **k: probe.read()
            )
            await rag.analyze_multimodal(
                doc_id=doc_id,
                file_path="demo.pdf",
                parsed_data=parsed_data,
                process_options="t",
            )
        finally:
            await rag.finalize_storages()
        peaks[item_count] = probe.peak
        # Without this the comparison below is vacuous: a backend stubbed out
        # entirely gives peaks {8: 0, 32: 0} and 0 <= 0 passes.
        assert probe.total == item_count, (
            f"no cache reads happened at item_count={item_count}; "
            f"this case is not measuring anything"
        )

    assert peaks[32] <= peaks[8], f"peak cache reads grew with document size: {peaks}"
