# BabbageBox Backend

This is the backend API for BabbageBox, a local AI assistant with LLM tool calling, TTS, memory, and search capabilities.

## Features

- Local LLM inference using llama-cpp-python with automatic tool calling
- Model-specific prompt formatting for Qwen, Gemma, TinyLlama, and Phi models
- Text-to-speech using Supertonic TTS
- Memory management with FAISS vector search
- MCP (Model Context Protocol) search integration (DuckDuckGo + Wikipedia)
- Conversation management
- Runtime model switching
- Audio transcription (stub)

## Prerequisites

- Python 3.8+
- pip

## Installation

1. Clone or navigate to the BabbageBox-Backend directory.

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # or
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables

Set the following environment variables (optional - if not set, sensible defaults will be auto-detected):

- `LLAMA_MODEL_PATH`: Path to your GGUF model file (auto-selects first available if not set)
- `LLAMA_CTX_SIZE`: Context size (default: 8192 tokens)
- `LLAMA_N_THREADS`: Number of CPU threads (default: auto-detected to match your CPU cores)

## Running the Server

Navigate to the Babbagebox-Backend directory, ensure the virtual environment is activated, then start the FastAPI server:

```bash
cd Babbagebox-Backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Endpoints

- `GET /api/health`: Health check
- `GET /api/config`: Get current configuration (model path)
- `POST /api/config`: Set runtime config (change model)
- `GET /api/models`: List available GGUF models
- `POST /api/process`: Process text with LLM (includes tool calling)
- `GET /api/personas`: List available personas
- `POST /api/tts`: Generate TTS audio
- `POST /api/supertonic`: Handle Supertonic TTS commands
- `POST /api/transcribe`: Transcribe audio (stub)
- `GET /api/memory`: List memories
- `POST /api/memory`: Create memory
- `PUT /api/memory/{id}`: Update memory
- `DELETE /api/memory/{id}`: Delete memory
- `POST /api/conversation/new`: Create new conversation
- `GET /api/conversations`: List all conversations
- `POST /api/mcp/search`: Search using MCP
- `POST /api/context/correct`: Correct context

## LLM Tool Calling

The backend supports automatic tool calling for LLMs. When processing text, the LLM can call search tools if needed:

- **Search Tool**: Queries DuckDuckGo and Wikipedia for real-time information
- Model-specific formatting: Different prompt structures for Qwen, Gemma, TinyLlama, etc.

## Text-to-Speech (TTS)

The backend supports high-quality TTS using the Supertonic engine:

- **Chat TTS**: Include `include_tts: true` in `/api/process` requests to generate audio for AI responses.
- **Direct TTS**: Use `/api/tts` endpoint with text to generate standalone speech audio.
- **Supertonic Commands**: Use `/api/supertonic` for command-based TTS (e.g., "say hello").

Audio is returned as base64-encoded WAV data.

## Model Management

- Place GGUF model files in the `../models/` directory
- Use `/api/config` to switch models at runtime without restarting
- Supported models: Qwen, Gemma, TinyLlama, Phi, and others

## Testing

Run tests with pytest:

```bash
pytest
```

## Data

Data is stored in the `data/` directory:
- `babbage.db`: SQLite database for conversations and memories