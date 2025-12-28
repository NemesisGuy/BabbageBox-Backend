# TinyLlama Prompt Config

TINYLLAMA_PROMPT_TEMPLATE = '''<|system|>
{system_prompt}
{history}
<|user|>
{user_text}
<|assistant|>
'''

TINYLLAMA_STOP_TOKENS = [
    "<|user|>",
    "<|system|>",
]


TINYLLAMA_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant running locally. "
    "If the user assigns you a name or role, treat it as persistent memory "
    "and always use it when asked about your identity."
)
