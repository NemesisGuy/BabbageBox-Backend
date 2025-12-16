# Inference runner for BabbageBox
from llama_cpp import Llama

def run_tinyllama_inference(llama_model, prompt, stop, max_tokens=256):
    """
    Run inference with TinyLlama using the provided prompt and stop tokens.
    """
    result = llama_model.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        stop=stop,
    )
    text = result.get("choices", [{}])[0].get("text", "")
    return text.strip() if text else "(empty response)"
