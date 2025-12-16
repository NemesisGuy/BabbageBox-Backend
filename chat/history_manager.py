# Chat history manager for BabbageBox

def append_message(history, role, content):
    """
    Append a message to the conversation history.
    history: list of dicts with 'role' and 'content'
    """
    history.append({'role': role, 'content': content})
    return history
