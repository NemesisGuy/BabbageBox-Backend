# ⚙️ BabbageBox Backend

> **The high-performance core of BabbageBox. Orchestrates GGUF inference, RAG memory, and multi-modal tool calling.**

---

## ✨ Key Features

- **🧩 Modular Chat Harness**: Centralized prompt templates in `app/chat/` for precise control over TinyLlama, Gemma, and Qwen models.
- **🚀 Agile Inference**: Optimized `llama-cpp-python` integration with auto-detecting performance profiles.
- **🧠 Vector Memory**: Long-term contextual memory using SQLite and FAISS.
- **🎙 Supertonic TTS**: Native integration for high-fidelity speech synthesis.
- **🔍 MCP Search**: Autonomous web search capabilities via DuckDuckGo and Wikipedia.

---

## 🛠 Installation

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
```

### 2. Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Engine

Start the FastAPI worker using [uvicorn](https://www.uvicorn.org/):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> [!TIP]
> Use the `--reload` flag during development for hot-reloading on code changes.

---

## 🧪 Testing & Verification

Comprehensive test suite covering prompt fidelity and API integrity:

```bash
pytest tests/test_api.py
pytest tests/test_gemma_harness.py
pytest tests/test_tinyllama_harness.py
```

---

## 📁 System Folders

- `/app/chat`: Unified location for model configurations and prompt templates.
- `/data`: SQLite database (`babbage.db`) and FAISS indexes.
- `/models`: (Symlinked/Root) GGUF model storage.

---

> [!IMPORTANT]
> This backend is designed for local-first privacy. Ensure your `LLAMA_MODEL_PATH` points to a valid GGUF file for inference.



 cd "C:\Users\Reign\Documents\Python Projects\BabbageBox"; & ".\.venv\Scripts\Activate.ps1"; cd Babbagebox-Backend; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reloadb