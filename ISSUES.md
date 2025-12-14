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