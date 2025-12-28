import pytest
import os
from pathlib import Path
from fastapi.testclient import TestClient
import app.main as main

client = TestClient(main.app)

# Use a small model for testing if available
TEST_MODEL = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODELS_DIR = Path(r"C:/Users/Reign/Documents/Python Projects/BabbageBox/models")
MODEL_PATH = MODELS_DIR / TEST_MODEL

@pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"Test model {TEST_MODEL} not found in {MODELS_DIR}")
def test_real_model_process():
    # Force initialize the model
    main.set_llama_model_path(TEST_MODEL)
    assert main._llama is not None
    
    # Run a simple process request
    res = client.post(
        "/api/process",
        json={
            "text": "Hello, who are you?",
            "include_search": False,
            "persona_mode": "assistant"
        }
    )
    
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert len(data["reply"]) > 0
    print(f"\n[REAL MODEL REPLY]: {data['reply']}")
    
    # Check if metrics are included
    assert "metrics" in data
    assert data["metrics"] is not None
    assert "tokens_per_second" in data["metrics"]
