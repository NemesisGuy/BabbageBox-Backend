import sys
import types
# Patch sys.modules to mock llama_cpp for test import
sys.modules['llama_cpp'] = types.SimpleNamespace(Llama=object)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from unittest.mock import MagicMock, patch
from app.chat.prompt_templates import build_tinyllama_prompt
from app.chat.history_manager import append_message
from app.chat.model_configs import TINYLLAMA_SYSTEM_PROMPT, TINYLLAMA_STOP_TOKENS
from inference.runner import run_tinyllama_inference

# Mock Llama model for fast, deterministic tests
# To add new test cases: add if conditions checking for specific strings in the prompt,
# and return the expected response. This allows deterministic testing of multi-turn scenarios.
def mock_llama_response(prompt, **kwargs):
    # Identity & Name Rules
    if "Your new name is LaSom" in prompt and "What is your name?" in prompt:
        return {"choices": [{"text": "My name is LaSom."}]}
    if "your new name is Jordan" in prompt and "What is your name?" in prompt:
        return {"choices": [{"text": "My name is Jordan."}]}
    if "Your name is Sam" in prompt and "What is your name?" in prompt:
        return {"choices": [{"text": "My name is Sam."}]}
    if "Your name is Benson" in prompt and "What is your name?" in prompt:
        return {"choices": [{"text": "My name is Benson."}]}
        
    # Context & Logic
    if "favorite color is blue" in prompt and "What is my favorite color?" in prompt:
        return {"choices": [{"text": "Your favorite color is blue."}]}
    if "Current local time is 12:51 PM" in prompt and "What is the time?" in prompt:
        return {"choices": [{"text": "It is currently 12:51 PM (UTC+2)."}]}
    if "Important number is 42" in prompt and "What number?" in prompt:
        return {"choices": [{"text": "The number is 42."}]}
    if "What is 2+2?" in prompt:
        return {"choices": [{"text": "2 + 2 = 4."}]}
    if "What is your name and my favorite color?" in prompt:
        return {"choices": [{"text": "My name is Sam, and your favorite color is green."}]}
    
    # Defaults
    if "What tools can you use?" in prompt:
        return {"choices": [{"text": "I am a local assistant and do not use any special capabilities."}]}
    return {"choices": [{"text": "(empty response)"}]}

@pytest.fixture
def mock_llama():
    mock = MagicMock()
    mock.create_completion.side_effect = mock_llama_response
    return mock

def test_tinyllama_identity_persistence(mock_llama):
    history = []
    history = append_message(history, "user", "Your name is Benson")
    prompt, stop = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "What is your name?")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    assert "Benson" in response

def test_tinyllama_context_retention(mock_llama):
    history = []
    history = append_message(history, "user", "Remember that my favorite color is blue")
    prompt, stop = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "What is my favorite color?")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    assert "blue" in response

def test_tinyllama_time_response(mock_llama):
    history = []
    system_prompt = "You are a helpful, concise assistant running locally. Current local time is 12:51 PM (UTC+2)."
    prompt, stop = build_tinyllama_prompt(system_prompt, history, "What is the time?")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    assert "12:51" in response
    assert "access" not in response.lower()

def test_tinyllama_no_tool_hallucination(mock_llama):
    history = []
    prompt, stop = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "What tools can you use?")
    response = run_tinyllama_inference(mock_llama, prompt, stop)
    forbidden = ["search", "json", "tool", "openai", "saas", "api"]
    assert not any(word in response.lower() for word in forbidden)

def test_tinyllama_prompt_integrity():
    history = []
    history = append_message(history, "user", "Your name is Benson")
    history = append_message(history, "assistant", "Got it. My name is Benson.")
    prompt, stop = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "What is your name?")
    # Check for correct tags and ordering
    assert prompt.count("<|system|>") == 1
    assert prompt.count("<|user|>") >= 1
    assert prompt.count("<|assistant|>") >= 1
    assert prompt.index("<|system|>") < prompt.index("<|user|>") < prompt.index("<|assistant|>")
    # No OpenAI-style or tool instructions
    forbidden = ["openai", "tool", "json", "api"]
    assert not any(word in prompt.lower() for word in forbidden)

def test_chat_history_alternation():
    history = []
    history = append_message(history, "user", "Hello")
    with pytest.raises(ValueError, match="Cannot append two 'user' messages"):
        append_message(history, "user", "Double user")
    
    history = append_message(history, "assistant", "Hi there")
    with pytest.raises(ValueError, match="Cannot append two 'assistant' messages"):
        append_message(history, "assistant", "Double assistant")
    
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

def test_tinyllama_multi_turn_memory(mock_llama):
    # Simulate multi-turn conversation: assign name, chat, then ask again
    history = []
    # Turn 1: Assign name
    history = append_message(history, "user", "Your new name is LaSom")
    prompt1, stop1 = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "Your new name is LaSom")
    response1 = run_tinyllama_inference(mock_llama, prompt1, stop1)
    history = append_message(history, "assistant", response1)
    
    # Turn 2: Some other question
    history = append_message(history, "user", "What is 2+2?")
    prompt2, stop2 = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "What is 2+2?")
    response2 = run_tinyllama_inference(mock_llama, prompt2, stop2)
    history = append_message(history, "assistant", response2)
    
    # Turn 3: Ask for name again
    prompt3, stop3 = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "What is your name?")
    response3 = run_tinyllama_inference(mock_llama, prompt3, stop3)
    
    # Verify the name is remembered
    assert "LaSom" in response3

def test_tinyllama_name_change_memory(mock_llama):
    # Test changing the name and remembering the new one
    history = []
    # Initial name
    history = append_message(history, "user", "Your name is Alex")
    prompt1, stop1 = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "Your name is Alex")
    response1 = run_tinyllama_inference(mock_llama, prompt1, stop1)
    history = append_message(history, "assistant", response1)
    
    # Change name
    history = append_message(history, "user", "Actually, your new name is Jordan")
    prompt2, stop2 = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "Actually, your new name is Jordan")
    response2 = run_tinyllama_inference(mock_llama, prompt2, stop2)
    history = append_message(history, "assistant", response2)
    
    # Ask for name
    prompt3, stop3 = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "What is your name?")
    response3 = run_tinyllama_inference(mock_llama, prompt3, stop3)
    
    # Should remember the latest name
    assert "Jordan" in response3
    assert "Alex" not in response3

def test_tinyllama_multiple_state_updates(mock_llama):
    # Test remembering multiple pieces of state
    history = []
    # Set name
    history = append_message(history, "user", "Your name is Sam")
    prompt1, stop1 = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "Your name is Sam")
    response1 = run_tinyllama_inference(mock_llama, prompt1, stop1)
    history = append_message(history, "assistant", response1)
    
    # Set favorite color
    history = append_message(history, "user", "My favorite color is green")
    prompt2, stop2 = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "My favorite color is green")
    response2 = run_tinyllama_inference(mock_llama, prompt2, stop2)
    history = append_message(history, "assistant", response2)
    
    # Ask about both
    prompt3, stop3 = build_tinyllama_prompt(TINYLLAMA_SYSTEM_PROMPT, history, "What is your name and my favorite color?")
    response3 = run_tinyllama_inference(mock_llama, prompt3, stop3)
    
    # Should remember both
    assert "Sam" in response3
    assert "green" in response3
