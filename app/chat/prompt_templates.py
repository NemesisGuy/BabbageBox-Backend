"""
Consolidated prompt templates and builders for BabbageBox.
Provides functions to build prompts for various model architectures.
"""
from typing import List, Dict, Tuple
from app.chat.model_configs import (
    TINYLLAMA_SYSTEM_PROMPT, TINYLLAMA_STOP_TOKENS,
    GEMMA_SYSTEM_PROMPT, GEMMA_STOP_TOKENS,
    QWEN_SYSTEM_PROMPT, QWEN_STOP_TOKENS,
    MISTRAL_SYSTEM_PROMPT, MISTRAL_STOP_TOKENS,
    PHI2_SYSTEM_PROMPT, PHI2_STOP_TOKENS
)

def build_tinyllama_prompt(system_prompt: str, history: List[Dict], user_text: str) -> Tuple[str, List[str]]:
    """Build a prompt for TinyLlama using <|system|>, <|user|>, <|assistant|> tokens."""
    if not system_prompt:
        system_prompt = TINYLLAMA_SYSTEM_PROMPT
    
    prompt = f"<|system|>\n{system_prompt}\n"
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    
    prompt += f"<|user|>\n{user_text}\n<|assistant|>\n"
    return prompt, TINYLLAMA_STOP_TOKENS

def build_gemma_prompt(system_prompt: str, history: List[Dict], user_text: str) -> Tuple[str, List[str]]:
    """Build a prompt for Gemma using <start_of_turn> tokens."""
    if not system_prompt:
        system_prompt = GEMMA_SYSTEM_PROMPT
    
    prompt = f"{system_prompt}\n"
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif role == "assistant":
            prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
            
    prompt += f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"
    return prompt, GEMMA_STOP_TOKENS

def build_qwen_prompt(system_prompt: str, history: List[Dict], user_text: str) -> Tuple[str, List[str]]:
    """Build a prompt for Qwen using <|im_start|> tags."""
    if not system_prompt:
        system_prompt = QWEN_SYSTEM_PROMPT
        
    prompt = ""
    if system_prompt:
        prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
    prompt += f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    return prompt, QWEN_STOP_TOKENS

def build_mistral_prompt(system_prompt: str, history: List[Dict], user_text: str) -> Tuple[str, List[str]]:
    """Build a prompt for Mistral using [INST] and [/INST] tags."""
    if not system_prompt:
        system_prompt = MISTRAL_SYSTEM_PROMPT
        
    prompt = f"<s>[INST] {system_prompt} "
    
    first = True
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            if first:
                prompt += f"{content} [/INST]"
                first = False
            else:
                prompt += f"<s>[INST] {content} [/INST]"
        elif role == "assistant":
            prompt += f" {content} </s>"
            
    if first:
         prompt += f"{user_text} [/INST]"
    else:
         prompt += f"[INST] {user_text} [/INST]"
         
    return prompt, MISTRAL_STOP_TOKENS

def build_phi2_prompt(system_prompt: str, history: List[Dict], user_text: str) -> Tuple[str, List[str]]:
    """Build a prompt for Phi-2 using Instruct: and Output: labels."""
    if not system_prompt:
        system_prompt = PHI2_SYSTEM_PROMPT
        
    prompt = f"{system_prompt}\n"
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            prompt += f"Instruct: {content}\n"
        elif role == "assistant":
            prompt += f"Output: {content}\n"
            
    prompt += f"Instruct: {user_text}\nOutput:"
    return prompt, PHI2_STOP_TOKENS

def get_chat_format(model_path: str) -> str:
    """Determine the appropriate llama-cpp-python chat_format for a model."""
    path_lower = model_path.lower()
    if "gemma" in path_lower:
        return "gemma"
    if "qwen" in path_lower:
        return "chatml"
    if "llama" in path_lower:
        # Check if it's Llama 2/3 or TinyLlama
        if "tinyllama" in path_lower:
            return "chatml" # TinyLlama often uses ChatML or similar tags, 
                          # but actually llama-cpp has a 'tinyllama' format too sometimes.
                          # Based on existing code, we were using manual.
            # return "tinyllama"
        return "llama-2"
    if "mistral" in path_lower:
        return "mistral-instruct"
    if "phi" in path_lower:
        return "phi-2"
    return None
