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
    if role not in ("system", "user", "assistant", "tool"):
        raise ValueError(f"Invalid role: {role}")

    if role in ("user", "assistant"):
        if history:
            last = history[-1]["role"]
            if last == role:
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
        role = msg.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue

        if role in ("user", "assistant"):
            if role == last_role:
                continue
            last_role = role

        clean.append(msg)

    return clean
