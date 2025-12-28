import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.chat.prompt_templates import build_mistral_prompt, MISTRAL_SYSTEM_PROMPT, MISTRAL_STOP_TOKENS
from app.chat.history_manager import append_message

@pytest.fixture
def mock_llama():
    return MagicMock()

def test_mistral_prompt_building():
    history = []
    user_text = "Hello Mistral"
    prompt, stop = build_mistral_prompt(None, history, user_text)
    
    assert "<s>[INST]" in prompt
    assert MISTRAL_SYSTEM_PROMPT in prompt
    assert user_text in prompt
    assert "[/INST]" in prompt
    assert stop == MISTRAL_STOP_TOKENS

def test_mistral_multi_turn_prompt():
    history = [
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "I am Mistral."}
    ]
    user_text = "What can you do?"
    prompt, stop = build_mistral_prompt(None, history, user_text)
    
    assert "<s>[INST]" in prompt
    assert "What is your name? [/INST]" in prompt
    assert "I am Mistral. </s>" in prompt
    assert "[INST] What can you do? [/INST]" in prompt

def test_mistral_inference_fallback(mock_llama):
    from app.main import _generate_reply
    # Patch the global _llama in app.main
    with patch('app.main._llama', mock_llama):
        mock_llama.create_completion.return_value = {
            "choices": [{"text": "Hello from Mistral!"}]
        }
        
        reply = _generate_reply("<s>[INST] Hello [/INST]", stop=MISTRAL_STOP_TOKENS)
        assert reply == "Hello from Mistral!"
        mock_llama.create_completion.assert_called_once()
