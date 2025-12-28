"""
Consolidated model configurations for BabbageBox.
Contains system prompts and stop tokens for supported models.
"""

# TinyLlama
TINYLLAMA_SYSTEM_PROMPT = (
    "You are an AI assistant.\n"
    "Your name is Benson.\n"
    "You remember your name.\n"
    'When asked your name, you reply: "My name is Benson."'
)
TINYLLAMA_STOP_TOKENS = ["<|user|>", "<|system|>", "<|assistant|>", "</s>", "<|endoftext|>"]

# Gemma
GEMMA_SYSTEM_PROMPT = "You are a helpful AI assistant."
GEMMA_STOP_TOKENS = ["<end_of_turn>", "<eos>", "<bos>"]

# Qwen
QWEN_SYSTEM_PROMPT = "You are a helpful assistant."
QWEN_STOP_TOKENS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]

# Mistral 7B Instruct v0.2
MISTRAL_SYSTEM_PROMPT = "You are a helpful AI assistant."
MISTRAL_STOP_TOKENS = ["</s>", "[/INST]", "INST"]

# Phi-2 (Instruct format)
PHI2_SYSTEM_PROMPT = "Instruct: You are a helpful assistant.\n"
PHI2_STOP_TOKENS = ["\nOutput:", "Output:", "<|endoftext|>"]
