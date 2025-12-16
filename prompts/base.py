# Base prompt builder (for future models)
def build_base_prompt(system_prompt, history, user_text):
    """
    Build a generic prompt (override in model-specific prompt builders).
    """
    # This is a placeholder for future model configs
    return f"{system_prompt}\n{history}\n{user_text}", []
