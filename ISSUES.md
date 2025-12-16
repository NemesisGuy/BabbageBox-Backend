# 🐞 Bug: Chat Harness & Prompt Handling (TinyLlama, BabbageBox)

## Problem Analysis

### Observed Symptoms
- Model forgets assigned names after one turn
- Refuses to answer simple questions (e.g., time)
- Hallucinates tool instructions (e.g., search JSON)
- Behaves like a cloud/SaaS AI with guardrails
- Chat UI responses appear one message late

### Root Causes
- Incorrect or missing chat templates (TinyLlama requires <|system|>, <|user|>, <|assistant|> formatting)
- OpenAI-style prompts break behavior
- System prompt pollution (tool schemas, JSON, SaaS guardrails)
- Conversation state handling bugs (messages overwritten, not appended)
- Frontend/backend mismatch in conversation history
- Monolithic backend design (main.py tightly couples model, prompt, inference logic)

## Key Insight
Different models require different prompt formats, stop tokens, and system prompts. A single “universal” harness will cause failures. Each model must have its own config and prompt builder.

## Required Fixes

### 1. Model-Specific Configuration
- Create per-model configs (chat template, stop tokens, allowed system prompt style, context injection rules)
- Example: TinyLlamaConfig, LlamaInstructConfig, etc.
- Do NOT reuse OpenAI-style prompts for local models.

### 2. Prompt Handling Fixes
- Always use correct role tags for TinyLlama: <|system|>, <|user|>, <|assistant|>
- Strip all tool schemas, JSON, and SaaS guardrails
- Use minimal system prompt for TinyLlama: “You are a helpful, concise assistant running locally.”

### 3. Conversation State Integrity
- Ensure every user and assistant turn is appended, never overwritten
- Frontend must send full conversation history or a conversation_id resolvable by the backend
- Fix delayed-render bug (response appearing one message late)

### 4. Time & Context Injection
- If time is needed, inject explicitly via system prompt
- Do not expect the model to infer runtime context

### 5. Architectural Refactor
- Break main.py into small, reusable components:
	- models/ (tinyllama.py, llama_instruct.py, ...)
	- prompts/ (base.py, tinyllama_prompt.py, ...)
	- chat/ (history_manager.py, prompt_builder.py, ...)
	- inference/ (runner.py, ...)
- Each model must declare its own prompt format, allowed system prompt rules, and stop tokens

### 6. Documentation
- Update backend-api.md: document TinyLlama chat format, explain why OpenAI-style prompts are forbidden
- Add inline comments in prompt builder explaining role tag requirements and consequences of prompt pollution

## Success Criteria
- TinyLlama remembers names across turns
- No hallucinated tools or SaaS behavior
- Responses render immediately
- Prompt logic is reusable for future models
- Backend architecture is no longer monolithic

---

TinyLlama validated as reference implementation.

---

## Suggested Fix (Summary)
- Implement model-specific prompt configs and builders
- Refactor main.py into modular components
- Fix conversation state append logic
- Update documentation
- Verify with TinyLlama chat sanity test (name memory, time, identity)
# BabbageBox Backend Issues & Bug Tracker

## Known Issues

### Open Issues

#### TTS-001: TTS Fails After First Message
**Status:** Resolved  
**Priority:** High  
**Description:** Text-to-speech only worked for the first chat message, subsequent messages had null audio_base64.  
**Root Cause:** Supertonic TTS instance reuse issues.  
**Solution:** Modified to create new TTS instance per request with proper error handling.  
**Date Resolved:** 2025-12-11  
**Files Changed:** `app/main.py`, `src/components/ChatBox.vue`

#### LLM-001: Stub Responses Instead of Real LLM Output
**Status:** Resolved  
**Priority:** High  
**Description:** Backend returned placeholder responses instead of actual LLM inference.  
**Root Cause:** LLAMA_MODEL_PATH not set, no auto-selection of models.  
**Solution:** Added auto-selection of first available GGUF model from models/ directory.  
**Date Resolved:** 2025-12-11  
**Files Changed:** `app/main.py`, `README.md`

#### DOCS-001: Missing Backend Documentation
**Status:** Resolved  
**Priority:** Medium  
**Description:** No startup instructions or API documentation for backend.  
**Solution:** Created comprehensive README.md with installation, setup, and API docs.  
**Date Resolved:** 2025-12-11  
**Files Changed:** `README.md`

### Closed Issues

#### SETUP-001: Virtual Environment Not Documented
**Status:** Closed  
**Priority:** Low  
**Description:** Backend README didn't mention venv setup.  
**Solution:** Added venv creation and activation instructions.  
**Date Closed:** 2025-12-11

#### DEP-001: Missing Supertonic Dependency
**Status:** Closed  
**Priority:** Medium  
**Description:** Supertonic TTS imported but not in requirements.txt.  
**Solution:** Added supertonic to requirements.txt.  
**Date Closed:** 2025-12-11

## Feature Requests

### Planned Features

#### TTS-002: Voice Selection
**Status:** Pending  
**Priority:** Low  
**Description:** Allow users to select different TTS voices/styles.  
**Estimated Effort:** Medium

#### LLM-002: Model Switching at Runtime
**Status:** Pending  
**Priority:** Medium  
**Description:** Allow changing LLAMA_MODEL_PATH without restart.  
**Estimated Effort:** High

#### MEM-001: Memory Export/Import
**Status:** Pending  
**Priority:** Low  
**Description:** Add functionality to backup and restore conversation memories.  
**Estimated Effort:** Medium

## Testing Status

### Test Coverage
- Unit tests: Minimal (pytest setup exists)
- Integration tests: None
- API endpoint tests: Basic health check
- TTS functionality: Manually tested
- LLM inference: Manually tested with auto-model selection

### Known Test Gaps
- Memory vector search functionality
- MCP search integration
- Concurrent request handling
- Error recovery scenarios

## Performance Notes

### Current Performance
- Cold start: ~5-10 seconds (model loading)
- TTS generation: ~2-5 seconds per request
- LLM inference: ~1-3 seconds per response (depending on model size)
- Memory: ~100ms for vector search

### Performance Issues
- TTS generation is synchronous and blocks API responses
- No caching of embeddings
- Single-threaded LLM inference

## Security Considerations

### Current Security
- No authentication (local use only)
- CORS enabled for localhost
- No input validation beyond basic types
- SQLite database with no encryption

### Security Improvements Needed
- Input sanitization for LLM prompts
- Rate limiting for API endpoints
- Database encryption for sensitive data
- HTTPS support for remote access

## Maintenance Notes

### Dependencies
- FastAPI: Regularly updated
- llama-cpp-python: Tied to specific model formats
- Supertonic: New dependency, monitor for updates
- FAISS: CPU-only version, consider GPU support

### Database
- SQLite with auto-migration
- No backup strategy implemented
- Memory embeddings stored as BLOBs

### Monitoring
- Basic logging implemented
- No metrics collection
- No health monitoring beyond /api/health

---

## 🔧 Regression Fixes

### Frontend Model Name Display Regression (2025-12-15)
**Issue:** Frontend failed to update displayed model name when switching models via ModelSelector modal. Chat messages continued showing old model name (e.g., "qwen2.5-1.5b-instruct-q4_k_m") even after switching to new model (e.g., "gemma-2b").

**Root Cause:** `MainChat.vue` only fetched model name on component mount, not when model was changed. ModelSelector emitted `model-set` event but MainChat wasn't listening.

**Fix Applied:**
- Modified `MainChat.vue` to listen for `model-set` event from ModelSelector
- Added `onModelSet()` function that calls `fetchModel()` to update `modelName` ref
- Updated event handler in template: `@model-set="onModelSet"`

**Prevention:**
- Added unit test in `MainChat.spec.ts` to verify `onModelSet()` calls `fetchModel()`
- Test ensures regression doesn't reoccur by validating event handling

**Files Changed:**
- `Babbagebox-Frontend/src/components/MainChat.vue`
- `Babbagebox-Frontend/src/components/__tests__/MainChat.spec.ts`