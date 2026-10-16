"""A caller-supplied HTTP transport must survive LightRAG's failure handling.

``create_openai_async_client`` forwards ``client_configs`` into
``AsyncOpenAI(**merged_configs)``, so a caller configuring a proxy, an mTLS
transport, an instrumented transport or a shared connection pool injects *its
own* ``httpx.AsyncClient``. ``AsyncOpenAI.close()`` is
``await self._client.aclose()`` -- against that injected object -- and these
bindings closed the client on every failure branch and again in both success
``finally`` blocks. The tenacity decorators then re-entered with the same
kwargs, rebuilding an ``AsyncOpenAI`` around the already-closed transport, so
attempts 2 and 3 died locally with ``RuntimeError: Cannot send a request, as the
client has been closed.`` One transient blip became a permanent
``RetryError[APIConnectionError]``, the later attempts issued no HTTP request at
all, and the original cause was replaced by that bogus connection error.

Ownership is the rule: LightRAG closes what it created and leaves an injected
transport to its owner. The transport is recognised by *value*, not key
presence -- ``AsyncOpenAI`` reads ``http_client=None`` as "build your own" and
still installs its own ``AsyncHttpxClientWrapper``, which stays ours to close.

These tests drive the real OpenAI SDK over a loopback socket against a real
``httpx.AsyncClient``. No provider is stubbed and nothing about the transport is
faked: closing it is observable precisely because the client is real.
"""

import base64
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import numpy as np
import pytest
from tenacity import RetryError, wait_none

from lightrag.llm.openai import (
    _transport_is_caller_owned,
    openai_complete_if_cache,
    openai_embed,
)

pytestmark = pytest.mark.offline

CHAT_OK = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "SECOND"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

SERVER_ERROR = {"error": {"message": "transient blip", "type": "server_error"}}


# The embedding bindings are wrapped by ``wrap_embedding_func_with_attrs``,
# which declares a 1536-dimension output and rejects anything else, so the fake
# endpoint has to answer at that width for the test to isolate ownership.
EMBEDDING_DIM = 1536


def _embedding_ok(count: int = 1) -> dict:
    """Build a body in the base64 encoding the bindings request by default."""
    vectors = [
        base64.b64encode(
            np.full(EMBEDDING_DIM, i + 1, dtype=np.float32).tobytes()
        ).decode()
        for i in range(count)
    ]
    return {
        "object": "list",
        "model": "test-model",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


@dataclass
class _ServerState:
    """Per-path script of responses to serve, plus what actually arrived."""

    chat_responses: list[dict] = field(default_factory=list)
    embed_responses: list[dict] = field(default_factory=list)
    chat_requests: list[dict] = field(default_factory=list)
    embed_requests: list[dict] = field(default_factory=list)

    def take(self, script: list[dict], fallback: dict) -> dict:
        # Repeat the last scripted response once the script runs dry, so a test
        # that expected fewer attempts reports "extra request" instead of
        # blowing up inside the handler.
        return script.pop(0) if len(script) > 1 else (script[0] if script else fallback)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: _ServerState

    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}
        state = self.state

        if self.path.endswith("/chat/completions"):
            state.chat_requests.append(payload)
            body = state.take(state.chat_responses, SERVER_ERROR)
            status = 200 if body is CHAT_OK else 500
        elif self.path.endswith("/embeddings"):
            state.embed_requests.append(payload)
            body = state.take(state.embed_responses, SERVER_ERROR)
            status = 200 if body.get("object") == "list" else 500
        else:
            status, body = 404, {"error": {"message": "no such route"}}

        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def no_retry_wait(monkeypatch):
    """Strip the exponential backoff so the retrying cases stay fast."""
    monkeypatch.setattr(openai_complete_if_cache.retry, "wait", wait_none())
    monkeypatch.setattr(openai_embed.func.retry, "wait", wait_none())


@pytest.fixture
def server():
    """A loopback OpenAI-compatible endpoint driven by per-path response scripts."""
    state = _ServerState()
    handler = type("_BoundHandler", (_Handler,), {"state": state})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
async def injected_client():
    """A transport owned by the caller of the bindings.

    The teardown asserts the ownership rule holds even if a test body raised
    before reaching its own assertion, then closes it so the loop does not
    warn about a leaked connection pool.
    """
    client = httpx.AsyncClient()
    yield client
    was_closed = client.is_closed
    if not was_closed:
        await client.aclose()
    assert not was_closed, "LightRAG closed a caller-owned transport"


# ---------------------------------------------------------------------------
# Ownership detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transport_ownership_is_decided_by_value_not_key_presence():
    """``http_client=None`` still makes the SDK build a client we must close."""
    assert _transport_is_caller_owned(None) is False
    assert _transport_is_caller_owned({}) is False
    assert _transport_is_caller_owned({"http_client": None}) is False
    assert _transport_is_caller_owned({"max_retries": 2}) is False

    owned = httpx.AsyncClient()
    try:
        assert _transport_is_caller_owned({"http_client": owned}) is True
    finally:
        await owned.aclose()


@pytest.mark.asyncio
async def test_ownership_matches_the_sdk_for_a_falsy_client():
    """A falsy injected client is one the SDK already discarded.

    ``AsyncOpenAI`` assigns ``self._client = http_client or
    AsyncHttpxClientWrapper(...)`` -- truthiness, not ``is not None``. An
    injected client that is falsy is therefore thrown away by the SDK in favour
    of its own transport, and it is that SDK-built transport we are responsible
    for. A predicate testing ``is not None`` would call this caller-owned, skip
    the close, and leak the SDK's connection pool on every call.
    """

    class FalsyClient(httpx.AsyncClient):
        def __bool__(self) -> bool:
            return False

        def __len__(self) -> int:
            return 0

    falsy = FalsyClient()
    try:
        assert _transport_is_caller_owned({"http_client": falsy}) is False

        # Confirm the SDK really does discard it, so the predicate above is
        # tracking observed behaviour rather than our assumption about it.
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="test-key", http_client=falsy)
        try:
            assert client._client is not falsy
        finally:
            await client.close()
    finally:
        await falsy.aclose()


# ---------------------------------------------------------------------------
# Retry keeps working -- the fix proof
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_retry_keeps_the_injected_transport(
    server, injected_client, no_retry_wait
):
    """One 500, then success: the second attempt must reach the network."""
    state, base_url = server
    state.chat_responses = [SERVER_ERROR, CHAT_OK]

    result = await openai_complete_if_cache(
        model="test-model",
        prompt="hello",
        api_key="test-key",
        base_url=base_url,
        openai_client_configs={"http_client": injected_client},
    )

    assert result == "SECOND"
    # The retry is a real second request, not a local failure on a dead client.
    assert len(state.chat_requests) == 2


@pytest.mark.asyncio
async def test_embed_retry_keeps_the_injected_transport(
    server, injected_client, no_retry_wait
):
    """Same guarantee on the embedding path, which closed via ``async with``."""
    state, base_url = server
    state.embed_responses = [SERVER_ERROR, _embedding_ok()]

    result = await openai_embed(
        texts=["hello"],
        model="test-model",
        api_key="test-key",
        base_url=base_url,
        client_configs={"http_client": injected_client},
    )

    assert result.shape == (1, EMBEDDING_DIM)
    assert len(state.embed_requests) == 2


# ---------------------------------------------------------------------------
# The transport outlives a single call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_complete_does_not_break_the_next_call(
    server, injected_client, no_retry_wait
):
    """A call that exhausts its retries must not poison later calls.

    Before the fix the first call closed the transport on its first failure, so
    the second call -- sharing the same injected client, the documented pooling
    pattern -- failed without issuing a request.
    """
    state, base_url = server
    state.chat_responses = [SERVER_ERROR, SERVER_ERROR, SERVER_ERROR, CHAT_OK]

    with pytest.raises(RetryError):
        await openai_complete_if_cache(
            model="test-model",
            prompt="first",
            api_key="test-key",
            base_url=base_url,
            openai_client_configs={"http_client": injected_client},
        )
    assert len(state.chat_requests) == 3

    result = await openai_complete_if_cache(
        model="test-model",
        prompt="second",
        api_key="test-key",
        base_url=base_url,
        openai_client_configs={"http_client": injected_client},
    )

    assert result == "SECOND"
    assert len(state.chat_requests) == 4


@pytest.mark.asyncio
async def test_successful_complete_leaves_the_injected_transport_open(
    server, injected_client
):
    """The success path closed the client in a ``finally``; it must not."""
    state, base_url = server
    state.chat_responses = [CHAT_OK]

    result = await openai_complete_if_cache(
        model="test-model",
        prompt="hello",
        api_key="test-key",
        base_url=base_url,
        openai_client_configs={"http_client": injected_client},
    )

    assert result == "SECOND"
    assert injected_client.is_closed is False


@pytest.mark.asyncio
async def test_successful_embed_leaves_the_injected_transport_open(
    server, injected_client
):
    """``async with openai_async_client`` closed the transport on success too."""
    state, base_url = server
    state.embed_responses = [_embedding_ok()]

    await openai_embed(
        texts=["hello"],
        model="test-model",
        api_key="test-key",
        base_url=base_url,
        client_configs={"http_client": injected_client},
    )

    assert injected_client.is_closed is False


# ---------------------------------------------------------------------------
# The SDK-owned path keeps working
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_none_transport_still_takes_the_sdk_owned_path(
    server, injected_client
):
    """``http_client=None`` means "SDK builds one" -- that one is ours to close.

    Ownership detection must read the value, or this config would silently
    become caller-owned and leak its connection pool on every call.
    """
    state, base_url = server
    state.chat_responses = [CHAT_OK, CHAT_OK]

    for _ in range(2):
        result = await openai_complete_if_cache(
            model="test-model",
            prompt="hello",
            api_key="test-key",
            base_url=base_url,
            openai_client_configs={"http_client": None},
        )
        assert result == "SECOND"

    assert len(state.chat_requests) == 2
    assert _transport_is_caller_owned({"http_client": None}) is False
