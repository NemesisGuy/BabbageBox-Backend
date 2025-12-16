# Prompt builder for Gemma using the official instruct-tuned format.
from models.gemma import GEMMA_START_TOKEN, GEMMA_END_TOKEN, GEMMA_STOP_TOKENS, GEMMA_SYSTEM_PROMPT

def build_gemma_prompt(system_prompt, history, user_text):
    """
    Build a prompt for Gemma using the official token-based instruct format.
    This ensures clear separation of turns for the model.
    """
    # If no system prompt is provided, fall back to the default.
    if system_prompt is None:
        system_prompt = GEMMA_SYSTEM_PROMPT

    # The system prompt is the first part of the conversation, without tokens.
    prompt_parts = [system_prompt]

    # Add historical turns, formatting them with the correct roles and tokens.
    for msg in history:
        # Map our 'assistant' role to the 'model' role expected by Gemma.
        role = 'model' if msg['role'] == 'assistant' else 'user'
        prompt_parts.append(f"{GEMMA_START_TOKEN}{role}\n{msg['content']}{GEMMA_END_TOKEN}")

    # Add the current user's message.
    prompt_parts.append(f"{GEMMA_START_TOKEN}user\n{user_text}{GEMMA_END_TOKEN}")

    # Prime the model to generate its response.
    prompt_parts.append(f"{GEMMA_START_TOKEN}model\n")

    # Join all parts with a single newline.
    prompt = "\n".join(prompt_parts)

    return prompt, GEMMA_STOP_TOKENS