import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_llama_instance():
    mock = MagicMock()
    # Mock create_chat_completion to simulate coherence
    def chat_side_effect(messages, **kwargs):
        content = " ".join([m['content'] for m in messages if isinstance(m, dict)])
        if "What is your name?" in content:
            if "Your name is Benson" in content:
                return {"choices": [{"message": {"content": "My name is Benson."}}]}
            return {"choices": [{"message": {"content": "I am your AI assistant."}}]}
        if "Remember the number 99" in content and "What was the number?" in content:
            return {"choices": [{"message": {"content": "The number you asked me to remember was 99."}}]}
        return {"choices": [{"message": {"content": "I understand."}}]}
    
    # Mock create_completion as well just in case legacy fallback is hit
    def comp_side_effect(prompt, **kwargs):
        if "What is your name?" in prompt and "Benson" in prompt:
            return {"choices": [{"text": "My name is Benson."}]}
        if "What was the number?" in prompt and "99" in prompt:
            return {"choices": [{"text": "The number was 99."}]}
        return {"choices": [{"text": "I understand."}]}

    mock.create_chat_completion.side_effect = chat_side_effect
    mock.create_completion.side_effect = comp_side_effect
    return mock

@pytest.fixture
def client(mock_llama_instance, monkeypatch):
    # Patch llama before importing app.main
    with patch.dict('sys.modules', {'llama_cpp': MagicMock()}):
        import app.main
        # Direct monkeypatch of the class constructor to return our mock instance
        monkeypatch.setattr(app.main, "Llama", MagicMock(return_value=mock_llama_instance))
        with TestClient(app.main.app) as c:
            yield c

def test_multi_turn_coherence(client):
    # Turn 1: Identity
    resp1 = client.post("/api/process", json={
        "text": "Your name is Benson",
        "context": []
    })
    assert resp1.status_code == 200
    
    # Turn 2: Recall Identity
    resp2 = client.post("/api/process", json={
        "text": "What is your name?",
        "context": [
            {"role": "user", "content": "Your name is Benson"},
            {"role": "assistant", "content": "Understood."}
        ]
    })
    assert resp2.status_code == 200
    assert "Benson" in resp2.json()["reply"]
    
    # Turn 3: Memory persistence
    resp3 = client.post("/api/process", json={
        "text": "Remember the number 99",
        "context": [
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": "I am Benson."}
        ]
    })
    assert resp3.status_code == 200
    
    # Turn 4: Recall memory
    resp4 = client.post("/api/process", json={
        "text": "What was the number?",
        "context": [
            {"role": "user", "content": "Remember the number 99"},
            {"role": "assistant", "content": "OK."}
        ]
    })
    assert resp4.status_code == 200
    assert "99" in resp4.json()["reply"]
