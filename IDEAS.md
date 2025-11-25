# Quiz Solver Optimization Ideas

## ✅ COMPLETED OPTIMIZATIONS

### ✅ Phase 1: Dependency Graph Analysis (COMPLETED)
**Status**: ✅ Implemented in `parallel_executor.py`

**What was implemented**:
- Dependency graph builder analyzing task inputs/outputs
- Wave-based execution grouping independent tasks
- Dataframe dependency tracking

**Results**:
- Foundation for parallel execution established
- Tasks properly grouped into execution waves

---

### ✅ Phase 2: I/O Task Parallelization (COMPLETED)
**Status**: ✅ Implemented in `executor.py` lines 48-138

**What was implemented**:
- Detection of I/O-bound tools (download_file, fetch_text, render_js_page, etc.)
- Async parallel execution using `asyncio.gather()` for I/O waves
- Sequential fallback for mixed or single-task waves

**Results**:
- demo quiz: 12.2s (was 13.3s) → **1.1s improvement**
- demo-scrape: ~20s (similar)
- demo-audio: Reduced from initial state
- Total: 71-76s (was 85s originally) → **~14 second improvement (16% faster)**

---

### ✅ Phase 3: Full Parallelization with CPU-Bound Operations (COMPLETED)
**Status**: ✅ Implemented in `parallel_executor.py` + `tool_executors.py`

**What was implemented**:
1. **Tool Categorization**:
   - I/O-bound tools: download, fetch, scrape, API calls → async parallel
   - CPU-bound tools: parse_csv, calculate_statistics, transcribe_audio → thread pool parallel
   - Mixed waves → sequential (safe fallback)

2. **Thread Pool Execution**:
   - ThreadPoolExecutor with 4 workers for CPU-bound tasks
   - Prevents overwhelming system with too many concurrent operations
   - Proper async/sync bridging using `loop.run_in_executor()`

3. **Shared State Protection**:
   - Thread-safe locks (`_dataframe_lock`) for `dataframe_registry` access
   - All read operations use `.copy()` to prevent race conditions
   - All write operations wrapped in `with _dataframe_lock:` blocks

4. **Updated Files**:
   - `parallel_executor.py`: Added CPU_BOUND_TOOLS, IO_BOUND_TOOLS, thread pool logic
   - `tool_executors.py`: Added `_dataframe_lock`, locked all registry operations
   - `executor.py`: Added lock to `calculate_statistics` registry read

**Expected Benefits**:
- Concurrent parsing of multiple CSV files
- Parallel statistics calculations on independent dataframes
- Safer concurrent execution with no race conditions
- Better CPU utilization on multi-core systems

---

## 🚀 PENDING Speed Optimizations

### ✅ 1. Smart Caching 🔥 HIGH IMPACT (COMPLETED)
**Current State**: Re-downloads files, re-transcribes audio, re-parses CSVs on every retry or similar quiz.

**Proposed**: Multi-layer caching strategy
```python
class QuizCache:
    file_cache = {}  # URL → downloaded content
    transcription_cache = {}  # audio_hash → transcription
    parse_cache = {}  # csv_hash → dataframe
    render_cache = {}  # url → rendered HTML
    
    def get_or_download(self, url):
        if url in self.file_cache:
            logger.info(f"[CACHE_HIT] File: {url}")
            return self.file_cache[url]
        result = download_file(url)
        self.file_cache[url] = result
        return result
```

**Cache Strategy**:
- **File downloads**: Cache by URL (1 hour TTL)
- **Transcriptions**: Cache by audio file hash (permanent - audio doesn't change)
- **Rendered pages**: Cache by URL (5 minutes TTL - may have dynamic content)
- **CSV parsing**: Cache by content hash (permanent)

**Expected Savings**:
- First attempt: 0 seconds (no cache)
- Retry attempts: 10-30 seconds (skip download + transcription + parsing)
- Similar quizzes: 5-15 seconds

**Implementation Complexity**: Low (simple dict-based cache with TTL)

---

#### **Implementation Plan**

**Phase 1: Core Cache Infrastructure** (30 minutes)
- Create `cache_manager.py` with `QuizCache` class
- Implement TTL-based expiration using timestamps
- Add thread-safe operations with `threading.Lock()`
- Track cache statistics (hits, misses, time saved)

**Core Components**:
```python
@dataclass
class CachedItem:
    content: Any
    timestamp: float
    size: int = 0

class QuizCache:
    def __init__(self):
        self.file_cache: Dict[str, CachedItem] = {}
        self.transcription_cache: Dict[str, CachedItem] = {}
        self.parse_cache: Dict[str, CachedItem] = {}
        self.render_cache: Dict[str, CachedItem] = {}
        self._cache_lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "saved_seconds": 0}
    
    def get_file(self, url: str, ttl: int = 3600) -> Optional[Dict]:
        # Check cache, validate TTL, return if valid
    
    def set_file(self, url: str, content: Dict, ttl: int = 3600):
        # Store with timestamp
```

**Phase 2: Content Hashing** (20 minutes)
- Add `hash_content()`: SHA256 for bytes
- Add `hash_file()`: Read file and compute hash
- Enable content-based caching for immutable data

```python
def hash_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def hash_file(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        return hash_content(f.read())
```

**Phase 3: Tool Integration** (45 minutes)

**Modify `tool_executors.py`**:
```python
from cache_manager import QuizCache

quiz_cache = QuizCache()

async def download_file(url: str) -> Dict[str, Any]:
    # Check cache
    cached = quiz_cache.get_file(url, ttl=3600)
    if cached:
        logger.info(f"[CACHE_HIT] File: {url}")
        quiz_cache.stats["hits"] += 1
        return cached
    
    # Cache miss - download
    quiz_cache.stats["misses"] += 1
    result = await _download_file_uncached(url)
    quiz_cache.set_file(url, result)
    return result

def parse_csv(path: str = None, url: str = None) -> Dict[str, Any]:
    # Hash CSV content for cache key
    if url:
        # Fetch content first to hash
        content = fetch_csv_content(url)
    else:
        with open(path, 'rb') as f:
            content = f.read()
    
    csv_hash = hash_content(content)
    cached = quiz_cache.get_parsed_csv(csv_hash)
    if cached:
        logger.info(f"[CACHE_HIT] CSV: {csv_hash[:8]}...")
        return cached
    
    # Parse and cache
    result = _parse_csv_uncached(content)
    quiz_cache.set_parsed_csv(csv_hash, result, ttl=None)  # Permanent
    return result
```

**Modify `tools.py`**:
```python
from cache_manager import quiz_cache, hash_file

async def transcribe_audio(audio_path: str, api_key: Optional[str] = None):
    # Hash audio file
    audio_hash = hash_file(audio_path)
    
    # Check cache (permanent for audio)
    cached = quiz_cache.get_transcription(audio_hash)
    if cached:
        logger.info(f"[CACHE_HIT] Transcription: {audio_hash[:8]}...")
        quiz_cache.stats["hits"] += 1
        return cached
    
    # Transcribe and cache
    quiz_cache.stats["misses"] += 1
    result = await _transcribe_audio_uncached(audio_path, api_key)
    quiz_cache.set_transcription(audio_hash, result, ttl=None)  # Permanent
    return result

async def render_page(url: str) -> Dict[str, Any]:
    # Check cache (5 min TTL for dynamic content)
    cached = quiz_cache.get_rendered_page(url, ttl=300)
    if cached:
        logger.info(f"[CACHE_HIT] Rendered page: {url}")
        return cached
    
    # Render and cache
    result = await _render_page_uncached(url)
    quiz_cache.set_rendered_page(url, result, ttl=300)
    return result
```

**Phase 4: Statistics & Logging** (15 minutes)
```python
def log_cache_stats():
    with quiz_cache._cache_lock:
        stats = quiz_cache.stats
        total = stats['hits'] + stats['misses']
        hit_rate = (stats['hits'] / total * 100) if total > 0 else 0
        
        logger.info(f"[CACHE_STATS] Hits: {stats['hits']}, Misses: {stats['misses']}, "
                    f"Hit Rate: {hit_rate:.1f}%, Time Saved: {stats['saved_seconds']:.1f}s")

# Add to executor.py at quiz end
log_cache_stats()
```

**Phase 5: Testing & Validation** (30 minutes)
1. **Cold start test**: Run quiz with empty cache (baseline)
2. **Warm cache test**: Retry quiz, verify cache hits
3. **TTL test**: Wait 1 hour, verify file re-download
4. **Thread safety test**: Concurrent cache access with parallel execution
5. **Statistics test**: Verify hit/miss counts accurate

**Success Criteria**:
- ✅ Cache hit rate > 60% on retries
- ✅ Time savings: 10-30 seconds per retry
- ✅ No race conditions with thread safety
- ✅ Same answers with/without cache
- ✅ Clear [CACHE_HIT] markers in logs

**Rollout Strategy**:
- **Phase 1** (Safe): Cache immutable operations only (files, transcriptions)
- **Phase 2** (Expand): Add parsing cache (CSV, JSON)
- **Phase 3** (Full): Cache dynamic content (rendered pages, APIs)
- **Rollback**: Keep `_uncached()` versions for easy revert

**File Changes**:
1. **NEW**: `cache_manager.py` (~200 lines)
2. **MODIFIED**: `tool_executors.py` (~10 changes)
3. **MODIFIED**: `tools.py` (~5 changes)
4. **MODIFIED**: `executor.py` (~2 changes for stats logging)

**Estimated Timeline**: ~2 hours total implementation

**Status**: ✅ **PARTIALLY COMPLETED** - Core caching working, CSV caching disabled temporarily

**Implementation Summary**:
- Created `cache_manager.py` with `QuizCache` class and content hashing utilities
- Integrated caching into `tool_executors.py`: ✅ `download_file()`, ✅ `render_page()`
- Integrated caching into `tools.py`: ✅ `transcribe_audio()` 
- Added cache statistics logging to `executor.py`
- Thread-safe operations with `threading.Lock()`
- TTL-based expiration (1hr files, 5min renders, permanent transcriptions)
- Hash-based caching for audio files

**What's Working (Verified in Test Run 2)**:
- ✅ File download cache: `[CACHE_HIT] File: https://...demo-audio.opus` (2s saved)
- ✅ Transcription cache: `[CACHE_HIT] Transcription: b463f591...` (5s saved)
- ✅ Render cache: `[CACHE_HIT] Rendered page: https://...demo-audio` (3s saved)
- ✅ Cache statistics tracking
- ✅ Thread-safe concurrent access

**Temporarily Disabled**:
- ❌ CSV parsing cache (dataframe registry restoration is complex - needs redesign)
  - Issue: Cached result contains df_key, but dataframe isn't in registry on cache hit
  - Solution needed: Serialize/deserialize actual dataframe or skip CSV caching

**Performance Impact (Test Run 2 - Retry Attempt)**:
- **Cache hits detected**: 3 major operations (file, transcription, render)
- **Estimated time saved**: ~10 seconds per retry
- **Hit rate**: ~40% (3 hits out of ~7 cacheable operations)

**Next Steps**:
1. Fix CSV caching by serializing dataframes with pickle or parquet
2. Run full test to measure total time savings
3. Consider caching LLM responses for deterministic queries

---

### 2. Faster Completion Detection 🟡 MEDIUM IMPACT
**Current State**: Checks completion after EVERY task iteration (demo-audio made 5 completion checks).

**Proposed**: Skip completion checks when outcome is obvious
```python
TERMINAL_OPERATIONS = [
    'calculate_statistics',
    'extract_patterns', 
    'analyze_image',
    'fetch_from_api',
]

NON_TERMINAL_OPERATIONS = [
    'download_file',  # Just downloaded, obviously incomplete
    'transcribe_audio',  # Just transcribed, need to process
    'parse_csv',  # Just parsed, need to analyze
]

def should_check_completion(last_task):
    # Skip check if last operation was non-terminal
    if last_task['tool_name'] in NON_TERMINAL_OPERATIONS:
        return False
    return True
```

**Expected Savings**: 
- 1-2 LLM calls per quiz (fast checks are instant, but avoided LLM calls save ~1s each)
- ~1-2 seconds per quiz

**Implementation Complexity**: Low (add skip logic to completion checker)

---

#### **Implementation Plan**

**Phase 1: Operation Classification** (15 minutes)
- Add operation categorization constants to `completion_checker.py`
- Define TERMINAL_OPERATIONS (completion likely)
- Define NON_TERMINAL_OPERATIONS (definitely incomplete)
- Define INTERMEDIATE_OPERATIONS (uncertain, need LLM check)

**Operation Categories**:
```python
# completion_checker.py

# GENERIC CLASSIFICATION RULES (works for any tool/quiz type)

def classify_operation(tool_name: str) -> str:
    """
    Classify operation based on generic semantic patterns.
    Returns: 'terminal', 'non_terminal', or 'intermediate'
    
    This uses semantic analysis, not hardcoded tool names.
    Works for ANY quiz type: data analysis, vision, APIs, scraping, etc.
    """
    tool_lower = tool_name.lower()
    
    # TERMINAL PATTERNS - Operations that produce final results
    terminal_patterns = [
        'calculate', 'compute', 'aggregate', 'analyze',  # Calculations
        'extract', 'get_value', 'find_answer',           # Extractions
        'summarize', 'conclude', 'finalize',             # Finalizations
        'visualize', 'plot', 'chart', 'graph',           # Visualizations
        'measure', 'count', 'sum', 'mean', 'stats',      # Statistical operations
        'ocr', 'vision', 'detect', 'recognize',          # Vision/image analysis
        'call_llm',                                       # LLM extraction (final step)
    ]
    
    # NON-TERMINAL PATTERNS - Operations that are preprocessing
    non_terminal_patterns = [
        'download', 'fetch', 'get', 'retrieve',          # Data acquisition
        'parse', 'read', 'load', 'open',                 # Data loading
        'render', 'request', 'scrape_page',              # Page rendering
        'transcribe', 'convert', 'decode',               # Format conversion
        'filter', 'select', 'transform', 'clean',        # Data preparation
        'split', 'join', 'merge', 'reshape',             # Data manipulation
    ]
    
    # Check terminal patterns
    if any(pattern in tool_lower for pattern in terminal_patterns):
        return 'terminal'
    
    # Check non-terminal patterns
    if any(pattern in tool_lower for pattern in non_terminal_patterns):
        return 'non_terminal'
    
    # Default to intermediate (uncertain - need LLM check)
    return 'intermediate'


# Optional: Explicit overrides for specific tools (edge cases only)
OPERATION_OVERRIDES = {
    # NON-TERMINAL: Data acquisition and preprocessing
    'render_js_page': 'non_terminal',         # Fetches HTML content
    'fetch_text': 'non_terminal',             # Fetches text from URL
    'fetch_from_api': 'non_terminal',         # Fetches API data
    'download_file': 'non_terminal',          # Downloads binary files
    'parse_csv': 'non_terminal',              # Parses CSV into DataFrame
    'parse_excel': 'non_terminal',            # Parses Excel into DataFrame
    'parse_json_file': 'non_terminal',        # Parses JSON into data
    'parse_html_tables': 'non_terminal',      # Extracts tables from HTML
    'parse_pdf_tables': 'non_terminal',       # Extracts tables from PDF
    'clean_text': 'non_terminal',             # Cleans/normalizes text
    'extract_patterns': 'non_terminal',       # Extracts patterns via regex
    'transform_data': 'non_terminal',         # Reshapes data (pivot/melt)
    'transcribe_audio': 'non_terminal',       # Audio to text
    'extract_audio_metadata': 'non_terminal', # Gets audio metadata
    
    # INTERMEDIATE: Could be preprocessing or final analysis
    'dataframe_ops': 'intermediate',          # Filter/sum/mean - varies by context
    'apply_ml_model': 'intermediate',         # ML model - varies by task
    'geospatial_analysis': 'intermediate',    # Spatial analysis - varies
    
    # TERMINAL: Produce final results
    'calculate_statistics': 'terminal',       # Computes final statistics
    'analyze_image': 'terminal',              # Vision analysis results
    'create_chart': 'terminal',               # Creates visualization
    'create_interactive_chart': 'terminal',   # Creates interactive viz
    'make_plot': 'terminal',                  # Creates plot
    'generate_narrative': 'terminal',         # Generates final narrative
    'call_llm': 'terminal',                   # LLM produces final answer
    'zip_base64': 'terminal',                 # Final packaging for submission
}

def get_operation_type(tool_name: str) -> str:
    """
    Get operation type with override support.
    Checks explicit overrides first, then uses semantic classification.
    """
    # Check explicit overrides
    if tool_name in OPERATION_OVERRIDES:
        return OPERATION_OVERRIDES[tool_name]
    
    # Use semantic classification
    return classify_operation(tool_name)
```

**Phase 2: Execution Log Enhancement** (15 minutes)
- Track last executed operation in iteration
- Store tool_name in execution context
- Pass last_task info to completion checker

**Modify `executor.py`**:
```python
async def execute_plan(...):
    # ... existing code ...
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"=== Iteration {iteration} ===")
        
        # Track last executed task in this wave
        last_executed_task = None
        
        # Execute tasks wave by wave
        for wave_idx, wave_tasks in enumerate(waves):
            # ... existing execution logic ...
            
            # Track last task executed
            if wave_tasks:
                last_executed_task = wave_tasks[-1]
        
        # ... artifact formatting ...
        
        # Check completion with last task info
        if page_data:
            completion_status = await check_plan_completion(
                plan, 
                artifacts, 
                page_data,
                last_executed_task=last_executed_task  # NEW PARAMETER
            )
```

**Phase 3: Smart Completion Check** (20 minutes)
- Add `should_check_completion()` function
- Implement skip logic based on operation type
- Add bypass tracking for statistics

**Modify `completion_checker.py`**:
```python
def should_check_completion(
    last_task: Optional[Dict[str, Any]], 
    artifacts: Dict[str, Any]
) -> tuple[bool, str]:
    """
    Determine if completion check is needed based on last operation.
    Uses GENERIC semantic classification - works for ANY tool/quiz type.
    
    Returns:
        tuple: (should_check: bool, reason: str)
    """
    # Always check if no task executed (edge case)
    if not last_task:
        return True, "no_task_info"
    
    tool_name = last_task.get("tool_name", "")
    
    # Get operation type using semantic classification
    op_type = get_operation_type(tool_name)
    
    # SKIP CHECK: Non-terminal operations (obviously incomplete)
    if op_type == 'non_terminal':
        logger.info(f"[COMPLETION_SKIP] Non-terminal operation '{tool_name}' - skipping check")
        return False, f"non_terminal_{tool_name}"
    
    # FORCE CHECK: Terminal operations (likely complete)
    if op_type == 'terminal':
        logger.info(f"[COMPLETION_CHECK] Terminal operation '{tool_name}' - checking completion")
        return True, f"terminal_{tool_name}"
    
    # SMART CHECK: Artifact-based detection (generic for all data types)
    # Check for ANY final result artifacts (not just statistics)
    final_result_indicators = [
        'statistics', 'result', 'answer', 'value', 'output',  # Generic results
        'vision_result', 'ocr_text', 'detected_',             # Vision/OCR
        'api_response', 'response_data',                      # API data
        'chart_path', 'plot_path', 'visualization',           # Visualizations
        'summary', 'conclusion', 'final_',                    # Summaries
    ]
    
    has_final_result = any(
        indicator in str(k).lower() or indicator in str(v)
        for k, v in artifacts.items()
        for indicator in final_result_indicators
    )
    
    if has_final_result:
        logger.info("[COMPLETION_CHECK] Final result artifact detected - checking completion")
        return True, "final_result_present"
    
    # SMART CHECK: Extracted values (generic pattern)
    has_extracted = any(k.startswith('extracted_') for k in artifacts.keys())
    if has_extracted:
        logger.info("[COMPLETION_CHECK] Extracted values found - checking completion")
        return True, "extracted_values_present"
    
    # DEFAULT: Check for intermediate operations (uncertain)
    logger.info(f"[COMPLETION_CHECK] Intermediate operation '{tool_name}' - checking")
    return True, f"intermediate_{tool_name}"


async def check_plan_completion(
    plan_obj: Dict[str, Any], 
    artifacts: Dict[str, Any], 
    page_data: Dict[str, Any],
    last_executed_task: Optional[Dict[str, Any]] = None  # NEW PARAMETER
) -> Dict[str, Any]:
    """Check if plan execution is complete - with smart skip logic"""
    
    # PHASE 3 OPTIMIZATION: Skip check if last operation was non-terminal
    should_check, skip_reason = should_check_completion(last_executed_task, artifacts)
    
    if not should_check:
        logger.info(f"[COMPLETION_SKIP] Skipping completion check - reason: {skip_reason}")
        return {
            "answer_ready": False,
            "needs_more_tasks": True,
            "reason": f"Skipped check after {skip_reason}",
            "recommended_next_action": "Continue execution",
            "fast_check": True,
            "check_skipped": True
        }
    
    # Proceed with normal completion checking
    logger.info(f"[COMPLETION_CHECK] Performing check - reason: {skip_reason}")
    
    # ... existing fast check logic ...
    fast_result = check_completion_fast(artifacts, page_text, transcription_text)
    
    # ... rest of existing logic ...
```

**Phase 4: Statistics Tracking** (10 minutes)
- Track skipped vs. performed checks
- Log savings from skipped LLM calls
- Add metrics to completion stats

**Add to `completion_checker.py`**:
```python
# Global statistics (in-memory)
completion_check_stats = {
    "checks_performed": 0,
    "checks_skipped": 0,
    "llm_calls_avoided": 0,
    "fast_checks_used": 0,
    "time_saved_seconds": 0.0
}

def log_completion_stats():
    """Log completion check statistics"""
    stats = completion_check_stats
    total = stats["checks_performed"] + stats["checks_skipped"]
    skip_rate = (stats["checks_skipped"] / total * 100) if total > 0 else 0
    
    logger.info(
        f"[COMPLETION_STATS] Checks: {stats['checks_performed']}, "
        f"Skipped: {stats['checks_skipped']}, Skip Rate: {skip_rate:.1f}%, "
        f"LLM Calls Avoided: {stats['llm_calls_avoided']}, "
        f"Time Saved: {stats['time_saved_seconds']:.1f}s"
    )
```

**Phase 5: Integration & Testing** (20 minutes)
1. Update executor.py to pass last_executed_task
2. Update all check_plan_completion calls
3. Test with demo quizzes
4. Verify skip logic working correctly
5. Measure time savings

**Success Criteria**:
- ✅ Skip checks after download_file, parse_csv, transcribe_audio
- ✅ Always check after calculate_statistics, call_llm
- ✅ Same quiz results (no regression)
- ✅ 40-60% reduction in completion checks
- ✅ 1-2 second savings per quiz
- ✅ Clear [COMPLETION_SKIP] markers in logs

**File Changes**:
1. **MODIFIED**: `completion_checker.py` (~50 lines added)
   - Add operation classification constants
   - Add `should_check_completion()` function
   - Update `check_plan_completion()` signature
   - Add statistics tracking

2. **MODIFIED**: `executor.py` (~10 lines changed)
   - Track last_executed_task in wave execution
   - Pass last_executed_task to check_plan_completion
   - Add completion stats logging at end

**Estimated Timeline**: ~1.5 hours total implementation

**Design Compliance**:
- ✅ Generic operation classification (not demo-specific)
- ✅ Works for all quiz types (data, images, APIs, scraping)
- ✅ Infrastructure optimization (not business logic)
- ✅ No hardcoded assumptions about workflows
- ✅ LLM still makes final decision when needed

**Rollback Strategy**:
- Keep original `check_plan_completion()` signature (last_executed_task optional)
- If `last_executed_task=None`, always perform check (backward compatible)
- Can disable by setting all operations to INTERMEDIATE_OPERATIONS

**Performance Impact Prediction**:

**Generic Classification Examples (works for ANY quiz type)**:

**Data Analysis Quiz** (5 completion checks):
1. ❌ After `download_file` → SKIP (semantic: "download" = non-terminal)
2. ❌ After `transcribe_audio` → SKIP (semantic: "transcribe" = non-terminal)
3. ❌ After `parse_csv` → SKIP (semantic: "parse" = non-terminal)
4. ❌ After `filter_dataframe` → SKIP (semantic: "filter" = non-terminal)
5. ✅ After `calculate_statistics` → CHECK (semantic: "calculate" = terminal)

**Vision/Image Quiz** (4 completion checks):
1. ❌ After `download_file` → SKIP (semantic: "download" = non-terminal)
2. ❌ After `render_page` → SKIP (semantic: "render" = non-terminal)
3. ❌ After `extract_image` → SKIP (semantic: "extract" + no result = non-terminal)
4. ✅ After `analyze_image` → CHECK (semantic: "analyze" = terminal)

**API/Scraping Quiz** (3 completion checks):
1. ❌ After `fetch_html` → SKIP (semantic: "fetch" = non-terminal)
2. ❌ After `scrape_page` → SKIP (semantic: "scrape" = non-terminal)
3. ✅ After `extract_final_value` → CHECK (semantic: "extract" + "final" = terminal)

**Result Across All Quiz Types**: ~60-80% reduction in completion checks
- Average checks reduced: 4-5 → 1-2 (60-80% reduction)
- LLM calls saved: 3-4 × 1s = 3-4 seconds per quiz
- Works for: data, images, PDFs, APIs, scraping, audio, video, etc.

**Expected Savings per Quiz (Generic)**:
- Optimistic: 3-4 seconds (4 LLM calls avoided)
- Realistic: 2-3 seconds (accounting for fast checks)
- Conservative: 1-2 seconds (some checks still needed)

**Key Advantage**: Semantic patterns work for tools we haven't seen yet!
- New tool `custom_aggregate_results()` → "aggregate" detected → terminal ✅
- New tool `fetch_external_data()` → "fetch" detected → non-terminal ✅
- No need to update code for new tools (generic by design)

---

## 🎯 Reliability Optimizations

### 6. Smarter Retry Strategy 🔥 HIGH IMPACT
**Current State**: Full retry from scratch on wrong answer - re-download, re-transcribe, re-parse everything.

**Proposed**: Partial retry - keep successful artifacts
```python
def retry_quiz(artifacts, wrong_answer):
    # Keep expensive preprocessing results
    kept_artifacts = {}
    for key, value in artifacts.items():
        # Keep: transcriptions, downloads, parsed data
        if any(prefix in key for prefix in ['transcribe_', 'download_', 'parse_', 'rendered_']):
            kept_artifacts[key] = value
            logger.info(f"[RETRY] Keeping artifact: {key}")
    
    # Only regenerate: plan, calculations, extractions
    return solve_with_artifacts(kept_artifacts)
```

**Example (demo-audio retry)**:
- **Current**: Re-download audio (3s) + re-transcribe (5s) + re-parse CSV (2s) + new calculation (2s) = 12s
- **Proposed**: Keep transcription + CSV, new calculation only = 2s
- **Savings**: 10 seconds per retry

**Implementation Complexity**: Low (filter artifacts before retry)

---

### 7. Confidence Scoring 🟡 MEDIUM IMPACT
**Current State**: Submit first answer found, even if uncertain.

**Proposed**: LLM returns confidence score with answer
```python
def extract_final_answer(artifacts):
    # Ask LLM: "What is the answer and how confident are you (0-1)?"
    result = llm.call({
        "extract_answer": True,
        "return_confidence": True
    })
    
    if result['confidence'] < 0.8:
        logger.warning(f"Low confidence ({result['confidence']}), trying alternative approach")
        return try_alternative_method(artifacts)
    
    return result['answer']
```

**Expected Impact**: 
- Fewer wrong submissions → fewer retries
- 10-20% reduction in total attempts

**Implementation Complexity**: Low (add confidence field to LLM prompt)

---

### 8. Multi-Path Execution 🟢 LOW IMPACT
**Current State**: Single execution path - if LLM chooses wrong approach, must retry from scratch.

**Proposed**: Generate multiple alternative plans, execute most promising
```python
def solve_quiz(page):
    plans = generate_alternative_plans(page, num_plans=3)
    # plans = [
    #   Plan A: transcribe → extract number,
    #   Plan B: transcribe → parse CSV → calculate,
    #   Plan C: download CSV directly → calculate
    # ]
    
    # Rank by confidence/simplicity
    best_plan = rank_plans(plans)[0]
    result = execute_plan(best_plan)
    
    if not result['correct'] and len(plans) > 1:
        # Try second-best plan
        result = execute_plan(plans[1])
```

**Expected Impact**: 
- Better handling of ambiguous instructions
- 5-10% improvement in first-attempt success rate

**Implementation Complexity**: High (need plan ranking, fallback logic)

---

### 9. Tool Success Rate Tracking 🟢 LOW IMPACT
**Proposed**: Track which tools succeed/fail for different data types
```python
tool_stats = {
    "transcribe_audio": {
        "opus": {"attempts": 100, "success": 95, "avg_time": 5.2},
        "mp3": {"attempts": 50, "success": 49, "avg_time": 4.8},
    },
    "call_llm": {
        "transcription": {"attempts": 20, "success": 4, "avg_time": 3.0},
    }
}

def select_tool(task_type, data_format):
    # Avoid tools with low success rates
    if tool_stats[tool][data_format]['success'] / tool_stats[tool][data_format]['attempts'] < 0.5:
        logger.warning(f"Tool {tool} has low success for {data_format}, trying alternative")
        return alternative_tool
```

**Expected Impact**: 
- Avoid repeatedly trying tools that historically fail
- 5% fewer failed operations

**Implementation Complexity**: Medium (persistent stats storage)

---

## 💰 Cost Optimizations

### 10. Smaller Model for Simple Tasks 🟡 MEDIUM IMPACT
**Current State**: gpt-4o-mini for ALL LLM calls (plan, completion check, next tasks).

**Proposed**: Use cheaper/faster models for simple tasks
```python
MODEL_SELECTION = {
    "plan_generation": "gpt-4o-mini",  # Complex reasoning needed
    "completion_check": "gpt-3.5-turbo",  # Simple yes/no
    "next_tasks": "gpt-4o-mini",  # Medium complexity
    "answer_extraction": "gpt-3.5-turbo",  # Pattern matching
}
```

**Cost Comparison** (per 1K tokens):
- gpt-4o-mini: $0.150 input / $0.600 output
- gpt-3.5-turbo: $0.50 input / $1.50 output (but faster)

**Expected Savings**: 
- 20-40% cost reduction on completion checks
- Faster response times for simple tasks

**Implementation Complexity**: Low (model selection logic)

---

### 11. Batched LLM Calls 🟢 LOW IMPACT
**Current State**: Separate calls for plan generation, completion check, next tasks.

**Proposed**: Combine when possible
```python
# Instead of:
# Call 1: "Generate plan"
# Call 2: "Is execution complete?"
# Call 3: "Generate next tasks"

# Do:
# Call 1: "Generate plan AND predict if you'll need multiple iterations"
# Call 2: "Check completion AND if incomplete, suggest next tasks"
```

**Expected Savings**: 
- 20-30% fewer API calls
- Reduced latency (fewer round trips)

**Implementation Complexity**: Medium (restructure prompts, handle combined outputs)

---

## 🧠 Generalization for Unknown Quizzes

### 12. Pattern Learning 🔥 HIGH IMPACT
**Proposed**: After solving each quiz, extract and store workflow pattern
```python
class WorkflowPattern:
    def __init__(self, quiz_type, steps):
        self.quiz_type = quiz_type
        self.steps = steps
        self.success_count = 0
        self.total_time = 0

patterns = {
    "js_scraping": WorkflowPattern(
        quiz_type="JavaScript Rendered Page",
        steps=[
            "fetch_text → detect <script> tag",
            "render_js_page → extract content",
            "extract_patterns → find answer"
        ]
    ),
    "audio_csv_analysis": WorkflowPattern(
        quiz_type="Audio Instructions + CSV Data",
        steps=[
            "download_file (audio)",
            "transcribe_audio → get instructions",
            "parse_csv → load data",
            "dataframe_ops (filter) → subset data",
            "calculate_statistics → compute answer"
        ]
    )
}

def recognize_pattern(page_text, artifacts):
    # Check if current quiz matches known pattern
    if "script" in page_text and ".js" in page_text:
        return patterns["js_scraping"]
    if "audio" in page_text and "csv" in page_text:
        return patterns["audio_csv_analysis"]
    return None

def solve_with_pattern(pattern):
    # Execute known-good workflow instead of exploring
    for step in pattern.steps:
        execute_step(step)
```

**Expected Impact**: 
- Solve similar quizzes 2-3x faster (skip exploration phase)
- Higher first-attempt success rate
- Self-improving system (learns from each quiz)

**Implementation Complexity**: High (pattern extraction, matching, storage)

---

### 13. Adaptive Forcing 🟡 MEDIUM IMPACT
**Current State**: Hardcoded force triggers (JS rendering when `<script>` detected, transcription for audio files, CSV parsing).

**Proposed**: Learn from failures and add new force patterns
```python
class AdaptiveForcingEngine:
    def __init__(self):
        self.force_rules = [
            # Initial hardcoded rules
            {"condition": "html_contains_script", "action": "render_js_page"},
            {"condition": "audio_file_present", "action": "transcribe_audio"},
        ]
    
    def learn_from_failure(self, quiz_state, correct_solution):
        # Analyze what was missing
        missing_step = find_missing_step(quiz_state, correct_solution)
        
        # Create new rule
        new_rule = {
            "condition": extract_condition(quiz_state),
            "action": missing_step
        }
        
        self.force_rules.append(new_rule)
        logger.info(f"[LEARNED] New forcing rule: {new_rule}")
```

**Example Learning**:
```
Quiz fails → Analysis shows we had PDF but never ran OCR
→ Learn: "if pdf_file_present and text_extraction_needed → force ocr_image"
→ Next PDF quiz: Automatically runs OCR without LLM deciding
```

**Expected Impact**: 
- Self-improving system
- Fewer exploration iterations on new quiz types

**Implementation Complexity**: High (failure analysis, rule extraction, persistent storage)

---

### 14. Hybrid Search Strategy 🟡 MEDIUM IMPACT
**Proposed**: Combine fast heuristics with LLM reasoning
```python
def solve_quiz_hybrid(page):
    # Phase 1: Fast heuristic matching (0.1s)
    pattern = match_known_pattern(page)
    if pattern and pattern.confidence > 0.9:
        logger.info(f"[FAST_PATH] High-confidence pattern match: {pattern.type}")
        return execute_pattern(pattern)
    
    # Phase 2: LLM exploration (slower but handles novel cases)
    logger.info("[SLOW_PATH] No pattern match, using LLM exploration")
    return llm_based_solve(page)
```

**Expected Impact**: 
- Best of both worlds: speed for known patterns, flexibility for novel quizzes
- 50%+ of quizzes solved via fast path

**Implementation Complexity**: Medium (pattern matching + fallback logic)

---

## 📊 Monitoring & Analytics

### 15. Performance Metrics Dashboard 🟢 LOW IMPACT
**Proposed**: Track detailed metrics for continuous improvement
```python
class QuizMetrics:
    def __init__(self):
        self.total_quizzes = 0
        self.success_rate = 0.0
        self.avg_time = 0.0
        self.avg_llm_calls = 0.0
        self.avg_cost = 0.0
        
        # Per-quiz-type breakdown
        self.by_type = {}
        
        # Tool performance
        self.tool_success_rates = {}
        self.tool_avg_times = {}
    
    def record_quiz(self, quiz_result):
        # Update all metrics
        # Identify bottlenecks
        # Suggest optimizations
```

**Tracked Metrics**:
- Success rate by quiz type
- Average time per quiz type
- LLM call count (where can we fast-check?)
- Cost per quiz
- Tool success rates
- Retry frequency

**Expected Impact**: 
- Data-driven optimization decisions
- Identify bottlenecks automatically

**Implementation Complexity**: Low (logging + aggregation)

---

## 🎯 Quick Wins (Priority Order)

### Immediate (This Session)
1. ✅ **Enhanced completion patterns** (DONE - saves 2-3 LLM calls)
2. ✅ **Transcription artifact selection** (DONE - fixes bug)
3. ✅ **Debug logging improvements** (DONE - better debugging)

### Next Session (Highest ROI)
4. **Smart caching** - 10-30 second savings on retries
5. **Smarter retry strategy** - Keep successful artifacts, 5-10 second savings
6. **Faster completion detection** - Skip obvious non-terminal operations

### Future (Medium ROI)
7. **Parallel task execution** - 5-10 second savings, complex implementation
8. **Confidence scoring** - Fewer wrong submissions
9. **Pattern learning** - 2-3x faster on similar quizzes

### Research (Long-term)
10. **Adaptive forcing** - Self-improving system
11. **Multi-path execution** - Better ambiguity handling
12. **Hybrid search** - Fast heuristics + LLM fallback

---

## 🔬 Experimental Ideas

### 16. Reinforcement Learning for Tool Selection
Train model to select optimal tools based on quiz characteristics.

### 17. Ensemble Solving
Run multiple solving strategies in parallel, submit most confident answer.

### 18. Meta-Learning
Learn how to learn - optimize the optimization process itself.

### 19. Cognitive Architecture
Model human problem-solving: understand → plan → execute → verify → adjust.

---

## 📝 Notes

**Design Principle Compliance**: All optimizations must maintain core principle: "Trust the LLM to interpret and decide." We optimize infrastructure, not force business logic.

**Testing Strategy**: Each optimization should be A/B tested on demo quizzes before deployment.

**Rollback Plan**: Keep old executor.py as reference, maintain compatibility during optimizations.
