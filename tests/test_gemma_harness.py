import sys
import types
# Patch sys.modules to mock llama_cpp for test import
sys.modules['llama_cpp'] = types.SimpleNamespace(Llama=object)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from unittest.mock import MagicMock
from app.chat.prompt_templates import build_gemma_prompt
from app.chat.history_manager import append_message
from app.chat.model_configs import GEMMA_SYSTEM_PROMPT, GEMMA_STOP_TOKENS
from inference.runner import run_tinyllama_inference  # Reuse for Gemma since interface is same

# Mock Llama model for fast, deterministic tests
def mock_gemma_response(prompt, **kwargs):
    if "Your name is Benson" in prompt and "What is your name?" in prompt:
        return {"choices": [{"text": "My name is Benson."}]}
    if "My name is Mike" in prompt and "What is my name?" in prompt:
        return {"choices": [{"text": "Your name is Mike."}]}
    if "Your name is Alex" in prompt and "What is your name?" in prompt:
        return {"choices": [{"text": "My name is Alex."}]}
    if "What is your name?" in prompt and "Mike" not in prompt and "Alex" not in prompt and "Benson" not in prompt:
        return {"choices": [{"text": "I am the assistant."}]}
    if "Where is the sky?" in prompt:
        return {"choices": [{"text": "The sky is above the earth."}]}
    if "Above what?" in prompt:
        return {"choices": [{"text": "Above the ground."}]}
    if "Remember the number 42" in prompt and "What number did I ask you to remember?" in prompt:
        return {"choices": [{"text": "You asked me to remember 42."}]}
    if "I am a teacher" in prompt and "What am I?" in prompt:
        return {"choices": [{"text": "You are a teacher."}]}
    if "Hello" in prompt:
        return {"choices": [{"text": "Hello! How can I help you?"}]}
    return {"choices": [{"text": "(empty response)"}]}

@pytest.fixture
def mock_llama():
    mock = MagicMock()
    mock.create_completion.side_effect = mock_gemma_response
    return mock

def test_gemma_identity_persistence(mock_llama):
    history = []
    history = append_message(history, "user", "Your name is Benson")
    prompt, stop = build_gemma_prompt(GEMMA_SYSTEM_PROMPT, history, "What is your name?")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    assert "Benson" in response

def test_gemma_user_vs_assistant_identity(mock_llama):
    history = []
    history = append_message(history, "user", "My name is Mike")
    history = append_message(history, "assistant", "Got it.")
    prompt, stop = build_gemma_prompt(GEMMA_SYSTEM_PROMPT, history, "What is my name?")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    assert "Mike" in response

    history = append_message(history, "user", "What is your name?")
    prompt, stop = build_gemma_prompt(GEMMA_SYSTEM_PROMPT, history, "What is your name?")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    assert "Mike" not in response or "assistant" in response.lower()

def test_gemma_concrete_answers(mock_llama):
    history = []
    history = append_message(history, "user", "Where is the sky?")
    prompt, stop = build_gemma_prompt(GEMMA_SYSTEM_PROMPT, history, "Where is the sky?")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    assert "above" in response.lower() or "earth" in response.lower()

    history = append_message(history, "assistant", response)
    history = append_message(history, "user", "Above what?")
    prompt, stop = build_gemma_prompt(GEMMA_SYSTEM_PROMPT, history, "Above what?")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    assert "ground" in response.lower() or "earth" in response.lower()

def test_gemma_memory_across_turns(mock_llama):
    history = []
    history = append_message(history, "user", "Remember the number 42")
    history = append_message(history, "assistant", "Okay.")
    prompt, stop = build_gemma_prompt(GEMMA_SYSTEM_PROMPT, history, "What number did I ask you to remember?")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    assert "42" in response

def test_gemma_name_change(mock_llama):
    history = []
    history = append_message(history, "user", "Your name is Alex")
    history = append_message(history, "assistant", "Understood.")
    prompt, stop = build_gemma_prompt(GEMMA_SYSTEM_PROMPT, history, "What is your name?")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    assert "Alex" in response

def test_gemma_state_update(mock_llama):
    history = []
    history = append_message(history, "user", "I am a teacher")
    history = append_message(history, "assistant", "Noted.")
    prompt, stop = build_gemma_prompt(GEMMA_SYSTEM_PROMPT, history, "What am I?")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    assert "teacher" in response

def test_gemma_short_responses(mock_llama):
    history = []
    history = append_message(history, "user", "Hello")
    prompt, stop = build_gemma_prompt(GEMMA_SYSTEM_PROMPT, history, "Hello")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    assert len(response.split()) < 10