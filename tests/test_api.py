import pytest
from fastapi.testclient import TestClient
from typing import List

import app.main as main


client = TestClient(main.app)


def test_personas_endpoint():
    res = client.get("/api/personas")
    assert res.status_code == 200
    data = res.json()
    assert "personas" in data
    assert "assistant" in data["personas"]



def get_mock_llama(captured=None):
    class MockLlama:
        def __call__(self, prompt, **kwargs):
            return {"choices": [{"text": "ok-via-call"}]}
        def create_chat_completion(self, messages, **kwargs):
            if captured is not None:
                captured.clear()
                captured.extend(messages)
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"completion_tokens": 10}
            }
        def create_completion(self, prompt, **kwargs):
            return {"choices": [{"text": "ok-via-completion"}]}
    return MockLlama()

def test_process_includes_search_context(monkeypatch):
    captured = []
    monkeypatch.setattr(main, "_llama", get_mock_llama(captured))
    monkeypatch.setattr(main, "mcp_search", lambda req: main.McpSearchResult(results=["result-one", "result-two"], providers=["mcp:search"]))
    monkeypatch.setattr(main, "_generate_reply", lambda prompt, **kwargs: "ok")

    res = client.post(
        "/api/process",
        json={
            "text": "who is luffy",
            "include_search": True,
            "conversation_id": None,
            "context": [],
            "persona_mode": "assistant",
            "custom_system_prompt": None,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "result-one" in data["context_used"]
    assert "mcp:search" in data["sources"]
    assert data["reply"] == "ok"


def test_process_without_search(monkeypatch):
    monkeypatch.setattr(main, "_llama", get_mock_llama())
    monkeypatch.setattr(main, "_generate_reply", lambda prompt, **kwargs: "ok-no-search")
    res = client.post(
        "/api/process",
        json={"text": "plain", "include_search": False},
    )
    assert res.status_code == 200
    data = res.json()
    assert "mcp:search" not in data["sources"]
    assert data["reply"] == "ok-no-search"


def test_process_generates_even_when_search_empty(monkeypatch):
    monkeypatch.setattr(main, "_llama", get_mock_llama())
    # Simulate search returning only placeholder text; with no other context, it should answer 'I don't know'.
    monkeypatch.setattr(
        main,
        "mcp_search",
        lambda req: main.McpSearchResult(results=["No direct answer found (DuckDuckGo/Wikipedia)."]),
    )

    monkeypatch.setattr(main, "_generate_reply", lambda prompt, stop=None, max_tokens=256: "I don't know based on the available context.")

    res = client.post(
        "/api/process",
        json={"text": "what is the meaning of 'bukkake'?", "include_search": True},
    )

    assert res.status_code == 200
    data = res.json()
    assert "No supporting info found from search." in data["context_used"]
    assert "mcp:search-empty" in data["sources"]
    assert data["reply"] == "I don't know based on the available context."


def test_mcp_search_normalizes_and_disables_safe_search(monkeypatch):
    captured = []

    class DummyResp:
        def __init__(self):
            self.status_code = 200
            self.ok = True

        def raise_for_status(self):
            return None

        def json(self):
            # Return a small abstract to stop after the first call.
            return {"AbstractText": "normalized hit"}

    def fake_get(url, params=None, timeout=None):
        captured.append({"url": url, "params": params, "timeout": timeout})
        return DummyResp()

    monkeypatch.setattr(main.requests, "get", fake_get)

    res = main.mcp_search(main.McpSearchRequest(query='what is the meaning of "bukkake"?'))

    assert res.results == ["normalized hit"]
    assert res.providers == ["ddg"]
    assert captured, "requests.get should be called"
    first_call = captured[0]
    assert first_call["params"]["q"] == "bukkake"
    assert first_call["params"]["kp"] == -2  # safe search off


def test_process_marks_search_error(monkeypatch):
    # Simulate search raising and ensure it marks search-error but still returns a response.
    def explode(req):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "_llama", get_mock_llama())
    monkeypatch.setattr(main, "mcp_search", explode)

    res = client.post(
        "/api/process",
        json={"text": "test", "include_search": True, "context": [{"role": "user", "content": "manual"}]},
    )

    assert res.status_code == 200
    data = res.json()
    assert "mcp:search-error" in data["sources"]
    assert data["reply"] == "ok"


def test_prompt_includes_guardrails(monkeypatch):
    captured = []
    monkeypatch.setattr(main, "_llama", get_mock_llama(captured))
    monkeypatch.setattr(main, "_generate_reply", lambda prompt, **kwargs: "ok")

    res = client.post(
        "/api/process",
        json={
            "text": "who are you?",
            "include_search": False,
            "context": [{"role": "user", "content": "memory1"}],
            "persona_mode": "storyteller",
        },
    )

    assert res.status_code == 200
    assert res.json()["reply"] == "ok"
    
    # Check roles strictly
    roles = [m.get("role") for m in captured]
    assert "system" in roles, f"System message missing in {roles}"
    assert "user" in roles, f"User message missing in {roles}"
    
    # Verify content
    messages = captured
    system_msg = next(m for m in messages if m["role"] == "system")
    assert "storyteller" in system_msg["content"] or "vivid storyteller" in system_msg["content"].lower()
    
    user_msgs = [m for m in messages if m["role"] == "user"]
    assert any("memory1" in m["content"] for m in user_msgs)
    assert user_msgs[-1]["content"] == "who are you?"


def test_faiss_dimension_mismatch_rebuilds(monkeypatch):
    # Simulate an index with the wrong dimension; ensure we rebuild and return no hits instead of asserting.
    class DummyBadIndex:
        def __init__(self, d):
            self.d = d

        def search(self, *args, **kwargs):
            raise AssertionError("dimension mismatch")

    class DummyGoodIndex:
        def __init__(self, d):
            self.d = d

        def search(self, query, k):
            import numpy as np

            scores = np.zeros((1, k), dtype="float32")
            idxs = -1 * np.ones((1, k), dtype="int64")
            return scores, idxs

    rebuild_called = {}

    def fake_rebuild():
        rebuild_called["count"] = rebuild_called.get("count", 0) + 1
        main._faiss_index = DummyGoodIndex(main.EMBED_DIM)
        main._mem_meta = [main.MemoryMeta(id=1, conversation_id=1)]

    monkeypatch.setattr(main, "_faiss_index", DummyBadIndex(d=16))
    monkeypatch.setattr(main, "_mem_meta", [main.MemoryMeta(id=1, conversation_id=1)])
    monkeypatch.setattr(main, "_rebuild_index", fake_rebuild)

    hits = main._search_memories_for_text("hi", conversation_id=1, top_k=5)

    assert hits == []
    assert rebuild_called.get("count", 0) == 1


def test_faiss_search_assertion_recovers(monkeypatch):
    # Simulate an index with correct dimension that still asserts during search; expect rebuild and empty hits.
    class DummyAssertIndex:
        def __init__(self, d):
            self.d = d

        def search(self, query, k):
            raise AssertionError("search failure")

    class DummyGoodIndex:
        def __init__(self, d):
            self.d = d

        def search(self, query, k):
            import numpy as np

            scores = np.zeros((1, k), dtype="float32")
            idxs = -1 * np.ones((1, k), dtype="int64")
            return scores, idxs

    rebuild_called = {}

    def fake_rebuild():
        rebuild_called["count"] = rebuild_called.get("count", 0) + 1
        main._faiss_index = DummyGoodIndex(main.EMBED_DIM)
        main._mem_meta = [main.MemoryMeta(id=1, conversation_id=1)]

    monkeypatch.setattr(main, "_faiss_index", DummyAssertIndex(d=main.EMBED_DIM))
    monkeypatch.setattr(main, "_mem_meta", [main.MemoryMeta(id=1, conversation_id=1)])
    monkeypatch.setattr(main, "_rebuild_index", fake_rebuild)

    hits = main._search_memories_for_text("hi", conversation_id=1, top_k=3)

    assert hits == []
    assert rebuild_called.get("count", 0) == 1


def test_rebuild_index_handles_mixed_embedding_shapes(monkeypatch):
    import numpy as np

    # Two rows: one 1D of correct size, one 2D (1,2048) to mimic llama output; both should normalize to EMBED_DIM.
    rows = [
        {"id": 1, "conversation_id": 1, "content": "a", "embedding": np.ones((32,), dtype=np.float32).tobytes()},
        {"id": 2, "conversation_id": 1, "content": "b", "embedding": np.ones((1, 2048), dtype=np.float32).tobytes()},
    ]

    class DummyCursor:
        def __init__(self, rows):
            self.rows = rows

        def fetchall(self):
            return self.rows

        def fetchone(self):  # Not used here
            return None

        def execute(self, *args, **kwargs):
            return self

    class DummyConn:
        def __init__(self, rows):
            self.rows = rows

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, *args, **kwargs):
            return DummyCursor(self.rows)

        def commit(self):
            return None

    monkeypatch.setattr(main, "_connect", lambda: DummyConn(rows))

    main._rebuild_index()

    assert main._faiss_index is not None
    assert getattr(main._faiss_index, "d") == main.EMBED_DIM
    assert len(main._mem_meta) == 2


def test_logs_endpoint():
    res = client.get("/api/logs")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    # Check if we have at least our startup logs or the test logs
    if len(data) > 0:
        assert "level" in data[0]
        assert "message" in data[0]
