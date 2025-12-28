import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.chat.prompt_templates import build_phi2_prompt, PHI2_SYSTEM_PROMPT, PHI2_STOP_TOKENS
from app.chat.history_manager import append_message

@pytest.fixture
def mock_llama():
    return MagicMock()

def test_phi2_prompt_building():
    history = []
    user_text = "Hello Phi-2"
    prompt, stop = build_phi2_prompt(None, history, user_text)
    
    assert PHI2_SYSTEM_PROMPT in prompt
    assert "Instruct: Hello Phi-2" in prompt
    assert "Output:" in prompt
    assert stop == PHI2_STOP_TOKENS

def test_phi2_multi_turn_prompt():
    history = [
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "I am Phi-2."}
    ]
    user_text = "What can you do?"
    prompt, stop = build_phi2_prompt(None, history, user_text)
    
    assert "Instruct: What is your name?" in prompt
    assert "Output: I am Phi-2." in prompt
    assert "Instruct: What can you do?" in prompt
    assert "Output:" in prompt

def test_phi2_inference_fallback(mock_llama):
    from app.main import _generate_reply
    # Patch the global _llama in app.main
    with patch('app.main._llama', mock_llama):
        mock_llama.create_completion.return_value = {
            "choices": [{"text": "Hello from Phi-2!"}]
        }
        
        reply = _generate_reply("Instruct: Hello\nOutput:", stop=PHI2_STOP_TOKENS)
        assert reply == "Hello from Phi-2!"
        mock_llama.create_completion.assert_called_once()
