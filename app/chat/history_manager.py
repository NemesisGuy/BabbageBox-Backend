"""
History manager for BabbageBox chat. Ensures strict role alternation and safe appending.
"""
from typing import List, Dict

def append_message(history: List[Dict], role: str, content: str) -> List[Dict]:
    """
    Append a message to the chat history, enforcing strict alternation:
    - Only 'user' or 'assistant' roles allowed
    - No duplicate roles in a row
    - Returns a new history list
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"Invalid role: {role}")
    if history and history[-1]["role"] == role:
        raise ValueError(f"Cannot append two '{role}' messages in a row.")
    return history + [{"role": role, "content": content}]


def sanitize_history(history: List[Dict]) -> List[Dict]:
    """
    Remove any invalid or duplicate role turns from history.
    Ensures alternation and only 'user'/'assistant' roles.
    """
    clean = []
    last_role = None
    for msg in history:
        if msg["role"] not in ("user", "assistant"):
            continue
        if msg["role"] == last_role:
            continue
        clean.append(msg)
        last_role = msg["role"]
    return clean
