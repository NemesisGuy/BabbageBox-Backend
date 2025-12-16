# Prompt builder for TinyLlama
from models.tinyllama import TINYLLAMA_PROMPT_TEMPLATE, TINYLLAMA_STOP_TOKENS, TINYLLAMA_SYSTEM_PROMPT

def build_tinyllama_prompt(system_prompt, history, user_text):
    """
    Build a prompt for TinyLlama using the correct chat template.
    history: list of dicts with 'role' and 'content'
    """
    history_str = ""
    for msg in history:
        if msg['role'] == 'user':
            history_str += f"<|user|>\n{msg['content']}<|endoftext|>\n"
        elif msg['role'] == 'assistant':
            history_str += f"<|assistant|>\n{msg['content']}<|endoftext|>\n"
    prompt = TINYLLAMA_PROMPT_TEMPLATE.format(system_prompt=system_prompt, history=history_str, user_text=user_text)
    return prompt, TINYLLAMA_STOP_TOKENS
