# Inference runner for BabbageBox
import logging

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore


def run_tinyllama_inference(llama_model, prompt, stop, max_tokens=256):
    """
    Run inference with TinyLlama using the provided prompt and stop tokens.
    If `llama_model` is None or inference fails, return a safe stub response
    instead of raising an exception to avoid 500 errors in the server.
    """
    if llama_model is None:
        logging.warning("run_tinyllama_inference called with no model; returning stub response")
        return "(model not available - stub response)"

    try:
        result = llama_model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            stop=stop,
        )
        text = result.get("choices", [{}])[0].get("text", "")
        return text.strip() if text else "(empty response)"
    except Exception as exc:
        logging.exception("TinyLlama inference failed: %s", exc)
        return "(inference failed - stub response)"
