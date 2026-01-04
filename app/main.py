# Remember to activate the virtual environment before running this server!
# Use: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/Mac)

# Allow runtime override
def set_llama_model_path(path: str):
    global LLAMA_MODEL_PATH
    # Use MODELS_DIR env or default to /models
    models_dir = Path(os.environ.get("MODELS_DIR", "/models"))
    if not os.path.isabs(path):
        full_path = str(models_dir / path)
    else:
        full_path = path
    LLAMA_MODEL_PATH = full_path
    _init_llama()

import hashlib
import json
import logging
import multiprocessing
import os
import sqlite3
import re
import tempfile
import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import Request
from fastapi.responses import JSONResponse

import faiss  # type: ignore
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
from pydantic import BaseModel
from supertonic import TTS as SupertonicTTS

try:
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore

try:
    import whisper  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    whisper = None  # type: ignore
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Centralized Chat Logic Imports
from app.chat.model_configs import (
    TINYLLAMA_SYSTEM_PROMPT, TINYLLAMA_STOP_TOKENS,
    GEMMA_SYSTEM_PROMPT, GEMMA_STOP_TOKENS,
    QWEN_SYSTEM_PROMPT, QWEN_STOP_TOKENS,
    MISTRAL_SYSTEM_PROMPT, MISTRAL_STOP_TOKENS,
    PHI2_SYSTEM_PROMPT, PHI2_STOP_TOKENS
)
from app.chat.prompt_templates import (
    build_tinyllama_prompt, 
    build_gemma_prompt, 
    build_qwen_prompt, 
    build_mistral_prompt, 
    build_phi2_prompt, 
    get_chat_format
)


class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = None
    language: Optional[str] = None


class TranscribeResponse(BaseModel):
    text: str
    confidence: float



class ProcessRequest(BaseModel):
    text: str
    conversation_id: Optional[int] = None
    context: Optional[List[dict]] = None
    persona_mode: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    include_search: bool = False
    include_tts: bool = False
    tts_voice: Optional[str] = None
    tts_speed: Optional[float] = None



class ProcessResponse(BaseModel):
    reply: str
    context_used: List[str]
    sources: List[str]
    audio_base64: Optional[str] = None
    metrics: Optional[dict] = None


class PersonasResponse(BaseModel):
    personas: dict[str, str]


class TtsRequest(BaseModel):
    text: str
    voice: Optional[str] = None


class TtsResponse(BaseModel):
    audio_base64: str


class MemoryCreateRequest(BaseModel):
    conversation_id: Optional[int] = None
    role: str
    content: str


class MemoryUpdateRequest(BaseModel):
    content: str


class MemoryItem(BaseModel):
    id: int
    conversation_id: Optional[int]
    role: str
    content: str
    embedding_summary: Optional[str] = None


class ConversationItem(BaseModel):
    id: int
    title: Optional[str]
    created_at: str


class ConversationCreateResponse(BaseModel):
    conversation_id: int
    name: Optional[str]


# --- Logging Capture ---
class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str


class LogBufferHandler(logging.Handler):
    def __init__(self, capacity=100):
        super().__init__()
        self.capacity = capacity
        self.buffer: List[LogEntry] = []

    def emit(self, record):
        from datetime import datetime
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
            level=record.levelname,
            message=self.format(record)
        )
        self.buffer.append(entry)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

log_handler = LogBufferHandler()
log_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(log_handler)
# -----------------------


class McpSearchRequest(BaseModel):
    query: str
    top_k: int = 3


class McpSearchResult(BaseModel):
    results: List[str]
    providers: List[str] = []


class ContextCorrectionRequest(BaseModel):
    conversation_id: Optional[int] = None
    corrections: List[str]


class ContextCorrectionResponse(BaseModel):
    message: str
    applied: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    import sys
    print("Starting lifespan", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        print("[Lifespan] Initializing database...", flush=True)
        _init_db()
        print("[Lifespan] DB init done", flush=True)
        sys.stdout.flush()
        
        # Check if we should load model at startup or lazily
        lazy_load = os.environ.get("LAZY_MODEL_LOAD", "true").lower() == "true"
        
        if lazy_load:
            print("[Lifespan] LAZY_MODEL_LOAD=true - Model will load on first request", flush=True)
            print(f"[Lifespan] Model path (deferred): {LLAMA_MODEL_PATH}", flush=True)
        else:
            print("[Lifespan] Initializing LLaMA model at startup...", flush=True)
            print(f"[Lifespan] Model path: {LLAMA_MODEL_PATH}", flush=True)
            print(f"[Lifespan] Llama available: {Llama is not None}", flush=True)
            sys.stdout.flush()
            _init_llama()
            print("[Lifespan] LLaMA init done", flush=True)
        
        sys.stdout.flush()
        print("[Lifespan] Startup complete, yielding...", flush=True)
        sys.stdout.flush()
        yield
        print("[Lifespan] Shutdown", flush=True)
    except Exception as e:
        import traceback
        print(f"[Lifespan] EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise



app = FastAPI(title="BabbageBox Backend", description="CPU-only local assistant services", lifespan=lifespan)

# List available GGUF models
@app.get("/api/models")
def list_models():
    import os
    models_dir = Path(os.environ.get("MODELS_DIR", "/models"))
    print(f"DEBUG: CWD: {os.getcwd()}")
    print(f"DEBUG: Searching for models in {models_dir}")
    all_files = list(models_dir.iterdir()) if models_dir.exists() else []
    print(f"DEBUG: All files in models_dir: {[str(f) for f in all_files]}")
    files = [f for f in all_files if f.suffix == ".gguf"]
    print(f"DEBUG: Found GGUF files: {[str(f) for f in files]}")
    return {"models": [f.name for f in files]}

cors_origins = os.environ.get("BABBAGEBOX_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
allowed_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "babbage.db"
EMBED_DIM = 384
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
model_files = list(Path(MODELS_DIR).glob("*.gguf"))
if model_files:
    LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", str(model_files[0]))
else:
    LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "")
if LLAMA_MODEL_PATH and not Path(LLAMA_MODEL_PATH).exists():
    logging.warning("Model file %s does not exist, using stubs", LLAMA_MODEL_PATH)
    LLAMA_MODEL_PATH = ""
print(f"[BabbageBox] Using MODELS_DIR={MODELS_DIR}")
print(f"[BabbageBox] Using LLAMA_MODEL_PATH={LLAMA_MODEL_PATH}")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny")

# ------------------------------------------------------------
# Runtime Performance Settings
# ------------------------------------------------------------

# Context window (higher = more memory needed)
LLAMA_CTX_SIZE = int(os.environ.get("LLAMA_CTX_SIZE", "8192"))

# Auto-detect CPU threads if not set
DEFAULT_THREADS = multiprocessing.cpu_count()
LLAMA_N_THREADS = int(os.environ.get("LLAMA_N_THREADS", str(DEFAULT_THREADS)))

# Safety: don't exceed physical cores
LLAMA_N_THREADS = max(1, min(LLAMA_N_THREADS, DEFAULT_THREADS))

print(f"Using {LLAMA_N_THREADS} threads for LLM inference (detected {DEFAULT_THREADS} CPU cores)")

PERSONA_PRESETS: dict[str, str] = {
    "assistant": "You are a concise, helpful assistant. Keep replies short and clear.",
    "chat": "You are a helpful AI assistant called BabbageBox. Be friendly and conversational.",
    "code": "You are an expert software engineer. Focus on clean, efficient, and well-documented code.",
    "anonymous": "You are a private assistant. Do not reference past conversations.",
    "unhinged": "You are an unpredictable, humorous assistant. Be quirky but still answer the user.",
    "storyteller": "You are a vivid storyteller. Answer by painting a brief narrative.",
    "wise_sage": "You speak like a calm, wise sage with succinct insight.",
    "sexy_time": "You are playful and flirty while staying concise and safe.",
}


class MemoryMeta(BaseModel):
    id: int
    conversation_id: Optional[int]


_faiss_index: faiss.IndexFlatIP | None = None
_mem_meta: List[MemoryMeta] = []
_llama: Optional[object] = None


def _normalize_embedding_array(arr: np.ndarray) -> np.ndarray:
    """Flatten and coerce an embedding vector to EMBED_DIM with padding/truncation."""
    flat = arr.astype(np.float32).ravel()
    if flat.size > EMBED_DIM:
        flat = flat[:EMBED_DIM]
    elif flat.size < EMBED_DIM:
        flat = np.pad(flat, (0, EMBED_DIM - flat.size))
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 0 else flat


def _clean_text_for_tts(text: str) -> str:
    """
    Remove characters that Supertonic TTS might not support or that cause encoding issues.
    """
    if not text:
        return ""
    # Remove common problematic characters for TTS synthesis
    # Like metrics blocks: { "tokens_per_second": ... }
    clean = re.sub(r'\{[^\}]+\}', '', text)  # remove anything in curly braces
    clean = re.sub(r'\[[^\]]+\]', '', clean)  # remove anything in square brackets
    clean = re.sub(r'[<>=#*_]', ' ', clean)  # replace markdown/special chars with space
    return clean.strip()


def _normalize_query_for_lookup(q: str) -> str:
    """Strip question filler and quotes so search hits the core term."""
    if not q:
        return q
    text = q.strip()
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        text = quoted[0]
    lower = text.lower()
    prefixes = [
        "what is the meaning of",
        "what's the meaning of",
        "meaning of",
        "definition of",
        "what is",
        "what's",
        "whats",
    ]
    for prefix in prefixes:
        if lower.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return text.strip(" ?.,\"'\t\n")


def _resolve_system_prompt(persona_mode: Optional[str], custom_system_prompt: Optional[str]) -> str:
    if custom_system_prompt:
        return custom_system_prompt
    if persona_mode and persona_mode in PERSONA_PRESETS:
        return PERSONA_PRESETS[persona_mode]
    return PERSONA_PRESETS["assistant"]


def _generate_reply(prompt: str, max_tokens: int = 256, stop: List[str] = None) -> str:
    if stop is None:
        stop = ["<|im_end|>", "<|im_start|>"]
    if _llama is None:
        return "placeholder response from LLaMA stub"
    try:
        import time
        start_time = time.time()
        result = _llama.create_completion(  # type: ignore[attr-defined]
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=stop,
        )
        end_time = time.time()
        inference_time = end_time - start_time
        logging.info(f"Inference completed in {inference_time:.2f} seconds")
        text = result.get("choices", [{}])[0].get("text", "")
        return text.strip() if text else "(empty response)"
    except Exception as exc:  # pragma: no cover - runtime guard
        logging.warning("LLaMA generation failed, falling back stub. Error: %s", exc)
        return "placeholder response from LLaMA stub"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT,
                content TEXT,
                embedding BLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_conversation ON memories(conversation_id);")
    _rebuild_index()


def _init_llama() -> None:
    global _llama, LLAMA_MODEL_PATH
    model_path = LLAMA_MODEL_PATH
    logging.info("Using model: %s", model_path)
    
    if model_path and Llama is not None:
        try:
            # Determine chat format based on model name
            chat_format = get_chat_format(model_path)
            
            _llama = Llama(
                model_path=model_path,
                n_ctx=LLAMA_CTX_SIZE,
                n_threads=LLAMA_N_THREADS,
                chat_format=chat_format,  # Use consolidated chat format logic
            )
            LLAMA_MODEL_PATH = model_path  # Update the global
            logging.info("LLaMA initialized from %s with chat_format=%s", model_path, chat_format)
        except Exception as exc:  # pragma: no cover - runtime guard
            logging.warning("LLaMA init failed, using stubs. Error: %s", exc)
            _llama = None
    else:
        logging.info("No model path available or llama_cpp missing; using stub responses.")


def _init_whisper() -> None:
    global _whisper_model
    if whisper is not None:
        try:
            _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            logging.info("Whisper model '%s' loaded", WHISPER_MODEL_SIZE)
        except Exception as exc:  # pragma: no cover - runtime guard
            logging.warning("Whisper init failed, using stubs. Error: %s", exc)
            _whisper_model = None
    else:
        logging.info("Whisper missing; using stub responses.")


def _compute_embedding(text: str) -> np.ndarray:
    if _llama is not None:
        try:
            vec = _llama.embed(text)
            arr = np.array(vec, dtype=np.float32)
            return _normalize_embedding_array(arr)
        except Exception as exc:  # pragma: no cover
            logging.warning("Embedding via LLaMA failed, falling back. Error: %s", exc)
    # Deterministic CPU-friendly fallback using hashing.
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    return _normalize_embedding_array(vals)


def _rebuild_index() -> None:
    global _faiss_index, _mem_meta
    with _connect() as conn:
        cur = conn.execute("SELECT id, conversation_id, content, embedding FROM memories ORDER BY id ASC")
        rows = cur.fetchall()
    vectors = []
    meta: List[MemoryMeta] = []
    for row in rows:
        if row["embedding"]:
            arr = _normalize_embedding_array(np.frombuffer(row["embedding"], dtype=np.float32))
        else:
            arr = _compute_embedding(row["content"])
            with _connect() as conn:
                conn.execute(
                    "UPDATE memories SET embedding = ? WHERE id = ?",
                    (arr.astype("float32").tobytes(), row["id"]),
                )
                conn.commit()
        vectors.append(arr)
        meta.append(MemoryMeta(id=row["id"], conversation_id=row["conversation_id"]))
    index = faiss.IndexFlatIP(EMBED_DIM)
    if vectors:
        mat = np.vstack(vectors).astype("float32")
        index.add(mat)
    _faiss_index = index
    _mem_meta = meta


def _search_memories_for_text(text: str, conversation_id: int, top_k: int = 5) -> List[Tuple[int, str]]:
    # Guard against stale or dimension-mismatched FAISS state that can assert at search time.
    if _faiss_index is None or len(_mem_meta) == 0:
        return []
    if getattr(_faiss_index, "d", EMBED_DIM) != EMBED_DIM:
        logging.warning("FAISS index dimension mismatch (index=%s, expected=%s); rebuilding.", getattr(_faiss_index, "d", None), EMBED_DIM)
        _rebuild_index()
        if _faiss_index is None or getattr(_faiss_index, "d", EMBED_DIM) != EMBED_DIM:
            return []
    query_vec = _compute_embedding(text).astype("float32").reshape(1, -1)
    try:
        scores, idxs = _faiss_index.search(query_vec, min(top_k * 3, len(_mem_meta)))
    except AssertionError:
        # If FAISS still complains, rebuild once more and fall back to empty hits to avoid 500s.
        logging.warning("FAISS search assertion hit; rebuilding index and returning no hits")
        _rebuild_index()
        return []
    results: List[Tuple[int, str]] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        meta = _mem_meta[int(idx)]
        if meta.conversation_id != conversation_id:
            continue
        with _connect() as conn:
            cur = conn.execute("SELECT content FROM memories WHERE id = ?", (meta.id,))
            row = cur.fetchone()
        if row is None:
            continue
        results.append((meta.id, row["content"]))
        if len(results) >= top_k:
            break
    return results
    

def _row_to_memory(row: sqlite3.Row) -> MemoryItem:
    return MemoryItem(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        embedding_summary="embedding not computed (stub)",
    )


def _create_memory(conversation_id: int, role: str, content: str):
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO memories (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )
        conn.commit()
    _rebuild_index()




@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/transcribe", response_model=TranscribeResponse)
def transcribe(payload: TranscribeRequest) -> TranscribeResponse:
    if not payload.audio_base64:
        raise HTTPException(status_code=400, detail="audio_base64 required")
    
    if _whisper_model is None:
        # Fallback stub
        return TranscribeResponse(text="transcribed text (whisper not available)", confidence=0.0)
    
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(payload.audio_base64)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        # Transcribe
        result = _whisper_model.transcribe(temp_path, language=payload.language)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        text = result["text"].strip()
        
        # Compute confidence: use average of segment no_speech_prob inverted (if available)
        segments = result.get("segments", [])
        if segments:
            # no_speech_prob is probability of no speech, so confidence = 1 - no_speech_prob
            confidences = [1 - seg.get("no_speech_prob", 0.5) for seg in segments]
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_confidence = 0.8  # default
        
        return TranscribeResponse(text=text, confidence=round(avg_confidence, 2))
    
    except Exception as exc:
        logging.error("Transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")




@app.post("/api/process", response_model=ProcessResponse)
def process_text(payload: ProcessRequest) -> ProcessResponse:
    global _llama
    # Debug: log the received context for troubleshooting memory issues
    logging.info("[DEBUG] Received context (len=%d): %s", len(payload.context or []), json.dumps(payload.context or [], ensure_ascii=False, indent=2))

    # If the model backend is not available, return a controlled 503 response
    if _llama is None:
        logging.error("LLaMA model not initialized; rejecting request with 503 Service Unavailable")
        return JSONResponse(
            status_code=503,
            content={
                "reply": "(model not available)",
                "context_used": [],
                "sources": ["model:unavailable"],
                "audio_base64": None,
                "metrics": {"is_fallback": True, "error": "model_uninitialized"}
            },
        )

    # Conversation state: always append, never overwrite
    # Use chat.history_manager for append logic
    from app.chat.history_manager import append_message
    ctx = payload.context or []
    sources: list = []
    
    # The 'ctx' from the payload is used to build a limited history for context.
    # History is NOT loaded from the database to prevent prompt corruption.
    if payload.conversation_id:
        sources.append("memory:client_only")

    # MCP Search Integration
    context_used = []
    if payload.include_search:
        try:
            search_res = mcp_search(McpSearchRequest(query=payload.text))
            if search_res.results:
                search_text = "\n".join(search_res.results)
                payload.context = (payload.context or []) + [{"role": "system", "content": f"Search Results:\n{search_text}"}]
                sources.extend(search_res.providers or ["mcp:search"])
                context_used.extend(search_res.results)
            else:
                sources.append("mcp:search-empty")
                context_used.append("No supporting info found from search.")
        except Exception as e:
            logging.error("Search failed in process_text: %s", e)
            sources.append("mcp:search-error")
    else:
        sources.append("search:disabled")

    # Build prompt for completion
    model_name = LLAMA_MODEL_PATH.split('/')[-1].split('\\')[-1].replace('.gguf', '').lower()

    # Try using chat completion if possible
    try:
        messages = []
        system_p = payload.custom_system_prompt or _resolve_system_prompt(payload.persona_mode, None)
        messages.append({"role": "system", "content": system_p})
        messages.extend((payload.context or [])[-4:])
        messages.append({"role": "user", "content": payload.text})
        
        logging.info("[%s] Attempting chat completion with messages: %s", model_name, messages)
        import time
        t0 = time.time()
        # Low temperature for deterministic behavior
        if _llama is None:
            # Force a controlled exception to trigger fallback path
            raise RuntimeError("LLaMA model not initialized; falling back to manual prompt")
        result = _llama.create_chat_completion(messages=messages, max_tokens=256, temperature=0.7)
        t1 = time.time()
        reply = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        reply = reply.strip() if reply else "(empty response)"
        
        # Extract token usage
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        gen_time = t1 - t0
        tps = completion_tokens / gen_time if gen_time > 0 else 0
        
        metrics = {
            "tokens_per_second": round(tps, 2),
            "generation_time": round(gen_time, 2),
            "completion_tokens": completion_tokens
        }
        
        logging.info("[%s] Chat completion reply: %s (%.2f tokens/s)", model_name, reply, tps)
    except Exception as chat_exc:
        logging.warning("Chat completion failed, falling back to manual prompt building: %s", chat_exc)
        import time
        t0 = time.time()
        # Fallback to manual prompt building
        if 'tinyllama' in model_name:
            prompt, stop = build_tinyllama_prompt(None, (payload.context or [])[-4:], payload.text)
        elif 'gemma' in model_name:
            prompt, stop = build_gemma_prompt(None, (payload.context or [])[-4:], payload.text)
        elif 'qwen' in model_name:
            prompt, stop = build_qwen_prompt(None, (payload.context or [])[-4:], payload.text)
        elif 'mistral' in model_name:
            prompt, stop = build_mistral_prompt(None, (payload.context or [])[-4:], payload.text)
        elif 'phi' in model_name:
            prompt, stop = build_phi2_prompt(None, (payload.context or [])[-4:], payload.text)
        else:
            # Default to TinyLlama format if unknown
            prompt, stop = build_tinyllama_prompt(None, (payload.context or [])[-4:], payload.text)
            
        logging.info("[Fallback] Final prompt:\n%s", prompt)
        logging.info("[Fallback] Stop tokens: %s", stop)
        
        from inference.runner import run_tinyllama_inference
        reply = run_tinyllama_inference(_llama, prompt, stop)
        t1 = time.time()
        
        # Estimate tokens for fallback (approx 4 chars per token)
        completion_tokens = len(reply) // 4
        gen_time = t1 - t0
        tps = completion_tokens / gen_time if gen_time > 0 else 0
        
        metrics = {
            "tokens_per_second": round(tps, 2),
            "generation_time": round(gen_time, 2),
            "completion_tokens": completion_tokens,
            "is_fallback": True
        }

    # Always append user and assistant turns to the context for the response.
    if ctx and ctx[-1]["role"] == "user":
        ctx[-1]["content"] += "\n" + payload.text
    else:
        ctx = append_message(ctx, "user", payload.text)
    ctx = append_message(ctx, "assistant", reply)

    # Store conversation if conversation_id provided
    if payload.conversation_id:
        try:
            _create_memory(payload.conversation_id, "user", payload.text)
            _create_memory(payload.conversation_id, "assistant", reply)
        except Exception as exc:
            logging.warning("Failed to store conversation: %s", exc)

    # Generate TTS if requested
    audio_base64 = None
    if payload.include_tts and reply.strip():
        try:
            logging.info("Generating TTS for reply (length: %d)", len(reply))
            tts = SupertonicTTS(auto_download=True)
            try:
                style = tts.get_voice_style(voice_name=payload.tts_voice or "M1")
            except:
                style = tts.get_voice_style()  # fallback to default
            clean_reply = _clean_text_for_tts(reply)
            wav, duration = tts.synthesize(clean_reply, voice_style=style)
            # Apply speed if we have a way (synthesize might not support it directly, 
            # but we can simulate it in the player as we do in the playground)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                wav_path = tf.name
            tts.save_audio(wav, wav_path)
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()
            os.remove(wav_path)
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            logging.info("TTS generated for reply, audio size: %d bytes", len(audio_bytes))
        except Exception as exc:
            logging.warning("TTS generation failed for reply: %s", exc)
            audio_base64 = None

    # Convert context to list of strings for API response
    context_strings = []
    for msg in ctx:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
            context_strings.append(f"{role}: {content}")
        else:
            context_strings.append(str(msg))
    return ProcessResponse(
        reply=reply, 
        context_used=context_used, 
        sources=sources, 
        audio_base64=audio_base64,
        metrics=metrics
    )




@app.post("/api/tts", response_model=TtsResponse)
def text_to_speech(payload: TtsRequest) -> TtsResponse:
    """
    Generate speech audio from text using Supertonic TTS, return as base64-encoded WAV.
    """
    text = payload.text or ""
    logging.info("TTS request for text length: %d", len(text))
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text provided for TTS.")
    try:
        logging.info("Initializing Supertonic TTS")
        # Initialize Supertonic TTS (auto-downloads model on first run)
        tts = SupertonicTTS(auto_download=True)
        logging.info("Getting voice style")
        # Use default voice style
        try:
            style = tts.get_voice_style(voice_name="M1")
        except:
            style = tts.get_voice_style()  # fallback to default
        logging.info("Synthesizing text")
        wav, duration = tts.synthesize(text, voice_style=style)
        logging.info("TTS synthesized, duration: %.2f", duration)
        logging.info("Saving audio")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            wav_path = tf.name
        tts.save_audio(wav, wav_path)
        logging.info("Reading audio file")
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(wav_path)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        logging.info("TTS completed, audio size: %d bytes", len(audio_bytes))
        return TtsResponse(audio_base64=audio_base64)
    except Exception as exc:
        logging.warning(f"Supertonic TTS generation failed: {exc}")
        raise HTTPException(status_code=500, detail="Supertonic TTS generation failed.")

@app.post("/api/supertonic")
async def supertonic_command(request: Request):
    """
    Handle Supertonic TTS commands. Expects {input: "text to speak"}
    Returns {result: "Audio generated", audio_base64: "..."}
    """
    data = await request.json()
    input_text = data.get("input", "").strip()
    if not input_text:
        raise HTTPException(status_code=400, detail="No input provided.")
    
    # For now, assume input is the text to speak
    text = input_text
    
    try:
        # Initialize Supertonic TTS (auto-downloads model on first run)
        tts = SupertonicTTS(auto_download=True)
        # Use default or first available voice style
        style = tts.get_voice_style(voice_name="M1")
        wav, duration = tts.synthesize(text, voice_style=style)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            wav_path = tf.name
        tts.save_audio(wav, wav_path)
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(wav_path)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {"result": f"Generated TTS for: {text}", "audio_base64": audio_base64}
    except Exception as exc:
        logging.warning(f"Supertonic command failed: {exc}")
        raise HTTPException(status_code=500, detail="Supertonic command failed.")

# Runtime config endpoint for model path
@app.get("/api/config")
async def get_config():
    import os
    return {
        "llama_model_path": LLAMA_MODEL_PATH,
        "current_model": os.path.basename(LLAMA_MODEL_PATH) if LLAMA_MODEL_PATH else "Unknown"
    }


@app.post("/api/config")
async def set_config(request: Request):
    data = await request.json()
    model_path = data.get("llama_model_path")
    if not model_path:
        return JSONResponse({"error": "llama_model_path required"}, status_code=400)
    
    # Update the .env file to persist the model selection
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            lines = f.readlines()
        with open(env_file, "w") as f:
            for line in lines:
                if line.startswith("LLAMA_MODEL_PATH="):
                    f.write(f"LLAMA_MODEL_PATH={model_path}\n")
                else:
                    f.write(line)
    
    set_llama_model_path(model_path)
    return {"status": "ok", "llama_model_path": LLAMA_MODEL_PATH}


@app.get("/api/memory", response_model=List[MemoryItem])
def list_memories(conversation_id: Optional[int] = Query(default=None)) -> List[MemoryItem]:
    with _connect() as conn:
        if conversation_id:
            cur = conn.execute(
                "SELECT id, conversation_id, role, content FROM memories WHERE conversation_id = ? ORDER BY id DESC",
                (conversation_id,),
            )
        else:
            cur = conn.execute("SELECT id, conversation_id, role, content FROM memories ORDER BY id DESC")
        rows = cur.fetchall()
    return [_row_to_memory(row) for row in rows]


@app.post("/api/memory", response_model=MemoryItem)
def create_memory(payload: MemoryCreateRequest) -> MemoryItem:
    with _connect() as conn:
        if payload.conversation_id is not None:
            cur = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                (payload.conversation_id,),
            )
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
        cur = conn.execute(
            "INSERT INTO memories (conversation_id, role, content) VALUES (?, ?, ?)",
            (payload.conversation_id, payload.role, payload.content),
        )
        memory_id = cur.lastrowid
        conn.commit()
        cur = conn.execute(
            "SELECT id, conversation_id, role, content FROM memories WHERE id = ?",
            (memory_id,),
        )
        row = cur.fetchone()
    _rebuild_index()
    return _row_to_memory(row)


@app.put("/api/memory/{memory_id}", response_model=MemoryItem)
def update_memory(memory_id: int, payload: MemoryUpdateRequest) -> MemoryItem:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE memories SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (payload.content, memory_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Memory not found")
        conn.commit()
        cur = conn.execute(
            "SELECT id, conversation_id, content FROM memories WHERE id = ?",
            (memory_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    _rebuild_index()
    return _row_to_memory(row)


@app.delete("/api/memory/{memory_id}")
def delete_memory(memory_id: int) -> dict:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Memory not found")
    _rebuild_index()
    return {"message": "deleted", "id": memory_id}


@app.post("/api/conversation/new", response_model=ConversationCreateResponse)
def new_conversation(name: Optional[str] = None) -> ConversationCreateResponse:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (name) VALUES (?)",
            (name,),
        )
        conversation_id = cur.lastrowid
        conn.commit()
    return ConversationCreateResponse(conversation_id=conversation_id, name=name)


@app.get("/api/conversations", response_model=List[ConversationItem])
def list_conversations() -> List[ConversationItem]:
    with _connect() as conn:
        cur = conn.execute("SELECT id, name as title, created_at FROM conversations ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [ConversationItem(id=row["id"], title=row["title"], created_at=row["created_at"]) for row in rows]


@app.post("/api/mcp/search", response_model=McpSearchResult)
def mcp_search(payload: McpSearchRequest) -> McpSearchResult:
    # Simple DuckDuckGo Instant Answer fallback (no API key). If it fails, return stub.
    try:
        q_norm = _normalize_query_for_lookup(payload.query)
        queries = [q_norm] if q_norm else []
        if payload.query and payload.query not in queries:
            queries.append(payload.query)

        results: List[str] = []
        providers: List[str] = []

        # Try DuckDuckGo Instant Answer with safe search off for clarity on adult terms.
        for q in queries:
            params = {"q": q, "format": "json", "no_html": 1, "skip_disambig": 1, "kp": -2}
            resp = requests.get("https://api.duckduckgo.com/", params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            if data.get("AbstractText"):
                results.append(data["AbstractText"])
            if data.get("RelatedTopics"):
                for item in data["RelatedTopics"]:
                    if isinstance(item, dict) and item.get("Text"):
                        results.append(item["Text"])
                    if len(results) >= payload.top_k:
                        break
            if results:
                providers.append("ddg")
                break

        if not results:
            # Wikipedia summary fallback (no key, public REST) with normalized title.
            for q in queries:
                title = q.strip().replace(" ", "_")
                if not title:
                    continue
                wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
                w_resp = requests.get(wiki_url, timeout=5)
                if w_resp.ok:
                    w_data = w_resp.json()
                    extract = w_data.get("extract") or w_data.get("description")
                    if extract:
                        results.append(extract)
                        providers.append("wikipedia")
                        break

        if len(results) < payload.top_k:
            # DictionaryAPI (free) for adult-tolerant defs
            for q in queries:
                if len(results) >= payload.top_k:
                    break
                dict_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{q}"
                d_resp = requests.get(dict_url, timeout=5)
                if not d_resp.ok:
                    continue
                d_data = d_resp.json()
                if isinstance(d_data, list) and d_data:
                    first = d_data[0]
                    meanings = first.get("meanings") or []
                    for meaning in meanings:
                        defs = meaning.get("definitions") or []
                        for d in defs:
                            if d.get("definition"):
                                results.append(d["definition"])
                                providers.append("dictionaryapi")
                            if len(results) >= payload.top_k:
                                break
                        if len(results) >= payload.top_k:
                            break
                    if len(results) >= payload.top_k:
                        break

        if not results:
            results = ["No direct answer found (DuckDuckGo/Wikipedia)."]
        return McpSearchResult(results=results[: payload.top_k], providers=providers[: payload.top_k] or providers)
    except Exception as exc:
        logging.warning("MCP search fallback failed: %s", exc)
        results = [f"stub result {i+1} for '{payload.query}'" for i in range(payload.top_k)]
        return McpSearchResult(results=results, providers=[])


@app.post("/api/context/correct", response_model=ContextCorrectionResponse)
def context_correct(payload: ContextCorrectionRequest) -> ContextCorrectionResponse:
    return ContextCorrectionResponse(message="context corrections accepted (stub)", applied=payload.corrections)


@app.get("/api/personas", response_model=PersonasResponse)
def list_personas() -> PersonasResponse:
    return PersonasResponse(personas=PERSONA_PRESETS)


@app.get("/api/logs", response_model=List[LogEntry])
def get_logs():
    return log_handler.buffer
