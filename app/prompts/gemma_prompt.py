# Prompt builder for Gemma
from models.gemma import GEMMA_SYSTEM_PROMPT, GEMMA_STOP_TOKENS

def build_gemma_prompt(system_prompt, history, user_text):
    """
    Build a prompt for Gemma using the chat template.
    history: list of dicts with 'role' and 'content'
    """
    prompt = f"{system_prompt}\n"
    for msg in history:
        if msg['role'] == 'user':
            prompt += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
        elif msg['role'] == 'assistant':
            prompt += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
    prompt += f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"
    return prompt, GEMMA_STOP_TOKENS