# Prompt builder for Qwen
from models.qwen import QWEN_PROMPT_TEMPLATE, QWEN_STOP_TOKENS, QWEN_SYSTEM_PROMPT

def build_qwen_prompt(system_prompt, history, user_text):
    """
    Build a prompt for Qwen using the correct chat template.
    history: list of dicts with 'role' and 'content'
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    
    prompt = ""
    for msg in messages:
        if msg['role'] == 'system':
            prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg['role'] == 'user':
            prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg['role'] == 'assistant':
            prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt, QWEN_STOP_TOKENS