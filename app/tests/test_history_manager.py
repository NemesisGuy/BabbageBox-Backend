import pytest
from app.chat.history_manager import append_message, sanitize_history

def test_append_message_strict_alternation():
    history = []
    history = append_message(history, "user", "hi")
    history = append_message(history, "assistant", "hello")
    history = append_message(history, "user", "how are you?")
    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you?"},
    ]
    # Cannot append same role twice
    with pytest.raises(ValueError):
        history = append_message(history, "user", "oops")
    with pytest.raises(ValueError):
        history = append_message(history[:-1], "assistant", "oops")
    # Only user/assistant allowed
    with pytest.raises(ValueError):
        append_message(history, "system", "bad")

def test_sanitize_history_removes_duplicates_and_invalid():
    messy = [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "oops"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "bad"},
        {"role": "assistant", "content": "again"},
        {"role": "user", "content": "how are you?"},
    ]
    clean = sanitize_history(messy)
    assert clean == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you?"},
    ]
