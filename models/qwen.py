# Qwen Prompt Config

QWEN_PROMPT_TEMPLATE = '''{system_prompt}
{history}
<|im_start|>user
{user_text}<|im_end|>
<|im_start|>assistant
'''

QWEN_STOP_TOKENS = ["<|im_end|>", "<|im_start|>"]

QWEN_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)