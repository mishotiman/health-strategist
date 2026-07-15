"""The opt-in LLM response cache — verified offline with a stub client so it
needs no API credits."""

import app.llm_cache as llm_cache


class _StubClient:
    """Minimal Anthropic-shaped client that counts calls and echoes a fixed text."""
    def __init__(self, text="hello"):
        self.calls = 0
        self._text = text
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        block = type("Block", (), {"type": "text", "text": self._text})()
        return type("Msg", (), {"content": [block]})()


_REQ = dict(model="m", max_tokens=8, messages=[{"role": "user", "content": "q"}])


def test_no_cache_when_dir_unset(monkeypatch):
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    client = _StubClient()
    assert llm_cache.complete_text(client, **_REQ) == "hello"
    llm_cache.complete_text(client, **_REQ)
    assert client.calls == 2  # every call hits the API


def test_second_call_is_served_from_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    client = _StubClient()
    first = llm_cache.complete_text(client, **_REQ)
    second = llm_cache.complete_text(client, **_REQ)  # identical request
    assert first == second == "hello"
    assert client.calls == 1  # second served from cache
    assert list(tmp_path.glob("*.json"))  # something was written


def test_different_request_misses(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    client = _StubClient()
    llm_cache.complete_text(client, **_REQ)
    other = dict(_REQ, messages=[{"role": "user", "content": "different"}])
    llm_cache.complete_text(client, **other)
    assert client.calls == 2  # different prompt → different key → regenerated
