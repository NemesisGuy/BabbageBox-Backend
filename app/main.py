# Remember to activate the virtual environment before running this server!
# Use: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/Mac)

# Allow runtime override
def set_llama_model_path(path: str):
    global LLAMA_MODEL_PATH
    # If only a filename is provided, prepend models directory
    if not os.path.isabs(path):
        models_dir = Path(r"C:/Users/Reign/Documents/Python Projects/BabbageBox/models")
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


class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = None
    language: Optional[str] = None


class TranscribeResponse(BaseModel):
    text: str
    confidence: float



class ProcessRequest(BaseModel):
    text: str
    conversation_id: Optional[int] = None
    context: Optional[List[str]] = None
    persona_mode: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    include_search: bool = False
    include_tts: bool = False



class ProcessResponse(BaseModel):
    reply: str
    context_used: List[str]
    sources: List[str]
    audio_base64: Optional[str] = None


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
    _init_db()
    _init_llama()
    _init_whisper()
    yield



app = FastAPI(title="BabbageBox Backend", description="CPU-only local assistant services", lifespan=lifespan)

# List available GGUF models
@app.get("/api/models")
def list_models():
    import os
    models_dir = Path(r"C:/Users/Reign/Documents/Python Projects/BabbageBox/models")
    print(f"DEBUG: CWD: {os.getcwd()}")
    print(f"DEBUG: Searching for models in {models_dir}")
    all_files = list(models_dir.iterdir()) if models_dir.exists() else []
    print(f"DEBUG: All files in models_dir: {[str(f) for f in all_files]}")
    files = [f for f in all_files if f.suffix == ".gguf"]
    print(f"DEBUG: Found GGUF files: {[str(f) for f in files]}")
    return {"models": [f.name for f in files]}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",  # match null and any host
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "babbage.db"
EMBED_DIM = 384
LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH")
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
    if not model_path:
        # Auto-select the first available GGUF model
        models_dir = Path(r"C:/Users/Reign/Documents/Python Projects/BabbageBox/models")
        if models_dir.exists():
            gguf_files = sorted(models_dir.glob("*.gguf"), key=lambda p: p.stat().st_size)
            if gguf_files:
                model_path = str(gguf_files[0])
                logging.info("Auto-selected smallest model: %s", model_path)
            else:
                logging.warning("No GGUF models found in %s", models_dir)
        else:
            logging.warning("Models directory not found: %s", models_dir)
    else:
        # If only a filename is provided, prepend models directory
        if not os.path.isabs(model_path):
            models_dir = Path(r"C:/Users/Reign/Documents/Python Projects/BabbageBox/models")
            model_path = str(models_dir / model_path)
    
    if model_path and Llama is not None:
        try:
            _llama = Llama(
                model_path=model_path,
                n_ctx=LLAMA_CTX_SIZE,
                n_threads=LLAMA_N_THREADS,
                embedding=True,
                chat_format=None,  # Disable automatic chat formatting to use manual prompts
            )
            LLAMA_MODEL_PATH = model_path  # Update the global
            logging.info("LLaMA initialized from %s", model_path)
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


def _build_prompt(model_name: str, system_prompt: str, ctx: list, user_text: str) -> Tuple[str, List[str]]:
    """Builds a model-specific prompt and returns the prompt and stop words."""
    
    conversation_history = ""
    if 'mistral' in model_name:
        conversation_parts = []
        for msg in ctx:
            if msg['role'] == 'user':
                conversation_parts.append(f"[INST] {msg['content']} [/INST]")
            elif msg['role'] == 'assistant':
                conversation_parts.append(f"{msg['content']}</s>")
        conversation_parts.append(f"[INST] {user_text} [/INST]")
        prompt = f"<s>{''.join(conversation_parts)}"
        stop = ["</s>"]
        return prompt, stop

    # For other models, build a simple string history
    for msg in ctx:
        conversation_history += f"<|{msg['role']}|>\n{msg['content']}<|endoftext|>\n"

    if 'qwen' in model_name:
        prompt = (
            f"<|im_start|>system\n{system_prompt}\n"
            f"{conversation_history}"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        stop = ["<|im_end|>", "<|im_start|>"]
    elif 'phi' in model_name:
        prompt = (
            f"System: {system_prompt}\n"
            f"{conversation_history}"
            f"User: {user_text}\n"
            "Assistant:"
        )
        stop = ["User:", "System:", "Assistant:"]
    elif 'gemma' in model_name:
        prompt = (
            f"<start_of_turn>user\n{system_prompt}\n\n"
            f"{conversation_history}{user_text}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        stop = ["<end_of_turn>"]
    else:
        # Default for TinyLlama
        prompt = (
            f"<|system|>\n{system_prompt}<|endoftext|>\n"
            f"{conversation_history}"
            f"<|user|>\n{user_text}<|endoftext|>\n"
            "<|assistant|>\n"
        )
        stop = ["<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>"]
        
    return prompt, stop

@app.post("/api/process", response_model=ProcessResponse)
def process_text(payload: ProcessRequest) -> ProcessResponse:
    ctx = payload.context or []
    sources: List[str] = []
    if payload.conversation_id:
        # Get recent conversation history instead of searching
        with _connect() as conn:
            cur = conn.execute(
                "SELECT role, content FROM memories WHERE conversation_id = ? ORDER BY id DESC LIMIT 20",
                (payload.conversation_id,),
            )
            recent_memories = cur.fetchall()
        # Reverse to get chronological order
        ctx.extend([{"role": row[0], "content": row[1]} for row in reversed(recent_memories)])
        sources.append("memory:recent")
    if payload.include_search:
        try:
            search_payload = McpSearchRequest(query=payload.text, top_k=3)
            search_res = mcp_search(search_payload)
            filtered_results = [
                item for item in (search_res.results or [])
                if not item.lower().startswith("no direct answer found")
                and not item.lower().startswith("stub result")
            ]
            if filtered_results:
                ctx.extend(filtered_results)
                if getattr(search_res, "providers", None):
                    sources.extend([f"mcp:{p}" for p in search_res.providers])
                else:
                    sources.append("mcp:search")
            else:
                ctx.append("No supporting info found from search.")
                sources.append("mcp:search-empty")
        except Exception as exc:
            logging.warning("Search inclusion failed: %s", exc)
            sources.append("mcp:search-error")
    else:
        sources.append("search:disabled")

    # Always generate a response - let the LLM decide if it needs external context

    model_name = LLAMA_MODEL_PATH.split('/')[-1].split('\\')[-1].replace('.gguf', '').lower()
    system_prompt = _resolve_system_prompt(payload.persona_mode, payload.custom_system_prompt)
    
    prompt, stop = _build_prompt(model_name, system_prompt, ctx, payload.text)

    reply = _generate_reply(prompt, stop=stop)
    
    # Check for tool call
    import json
    reply = reply.strip()
    if reply.endswith("<|im_end|>"):
        reply = reply[:-len("<|im_end|>")].strip()
    if reply.startswith("{") and reply.endswith("}"):
        try:
            tool_call = json.loads(reply)
            if "tool" in tool_call and tool_call["tool"] == "search" and "query" in tool_call:
                # Execute search
                search_payload = McpSearchRequest(query=tool_call["query"], top_k=3)
                search_res = mcp_search(search_payload)
                filtered_results = [
                    item for item in (search_res.results or [])
                    if not item.lower().startswith("no direct answer found")
                    and not item.lower().startswith("stub result")
                ]
                if filtered_results:
                    search_ctx = "\n".join(filtered_results)
                    sources.extend([f"mcp:{p}" for p in (search_res.providers or [])])
                else:
                    search_ctx = "No information found from search."
                    sources.append("mcp:search-empty")
                
                # Generate final response with search results
                followup_prompt, followup_stop = _build_prompt(model_name, system_prompt, search_ctx, payload.text)
                reply = _generate_reply(followup_prompt, stop=followup_stop)
        except json.JSONDecodeError:
            pass  # Not a tool call, use reply as is
    
    audio_base64 = None
    if getattr(payload, "include_tts", False):
        logging.info("Generating TTS for reply length: %d", len(reply))
        try:
            # Limit TTS text to 500 characters to avoid issues with long responses
            tts_text = reply[:500]
            tts_payload = TtsRequest(text=tts_text)
            tts_res = text_to_speech(tts_payload)
            audio_base64 = tts_res.audio_base64
            logging.info("TTS generated successfully")
        except Exception as exc:
            logging.warning("TTS call failed: %s", exc)
            audio_base64 = None

    # Store conversation if conversation_id provided
    if payload.conversation_id:
        try:
            _create_memory(payload.conversation_id, "user", payload.text)
            _create_memory(payload.conversation_id, "assistant", reply)
        except Exception as exc:
            logging.warning("Failed to store conversation: %s", exc)

    return ProcessResponse(reply=reply, context_used=ctx, sources=sources, audio_base64=audio_base64)




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
    return {"llama_model_path": LLAMA_MODEL_PATH}


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
