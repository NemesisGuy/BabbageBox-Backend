import pytest
from app.chat.history_manager import append_message

def test_consecutive_user_merge():
    # Simulate a chat context with a user turn
    ctx = []
    ctx = append_message(ctx, "user", "hi")
    # Simulate backend logic: consecutive user message should merge
    if ctx and ctx[-1]["role"] == "user":
        ctx[-1]["content"] += "\n" + "how are you?"
    else:
        ctx = append_message(ctx, "user", "how are you?")
    assert ctx == [
        {"role": "user", "content": "hi\nhow are you?"}
    ]
    # Now append an assistant reply
    ctx = append_message(ctx, "assistant", "I'm good!")
    assert ctx == [
        {"role": "user", "content": "hi\nhow are you?"},
        {"role": "assistant", "content": "I'm good!"}
    ]
    # Another user message (should not merge, alternation is restored)
    ctx = append_message(ctx, "user", "what's up?")
    assert ctx == [
        {"role": "user", "content": "hi\nhow are you?"},
        {"role": "assistant", "content": "I'm good!"},
        {"role": "user", "content": "what's up?"}
    ]
