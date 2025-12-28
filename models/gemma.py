# Gemma Prompt Config

GEMMA_PROMPT_TEMPLATE = '''{system_prompt}

{history}User: {user_text}

Assistant:'''

GEMMA_STOP_TOKENS = ["</s>"]

GEMMA_START_TOKEN = "<start_of_turn>"
GEMMA_END_TOKEN = "<end_of_turn>"

GEMMA_SYSTEM_PROMPT = (
    "You are Gemma, a helpful AI assistant created by Google."
)