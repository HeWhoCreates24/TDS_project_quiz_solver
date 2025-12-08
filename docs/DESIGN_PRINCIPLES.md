# Design Principles: Avoiding Overfitting

## Core Philosophy

**The system must work on UNSEEN test questions, not just the demo questions we have access to.**

This document outlines principles to ensure the quiz solver remains general-purpose and doesn't overfit to specific demo question patterns.

---

## ❌ What NOT to Do

### 1. **Never Hardcode Demo-Specific Logic**

```python
# ❌ BAD - Overfitted to demo-audio
if "cutoff" in page_text and "sum" in transcription:
    force_filter_and_sum()

# ✅ GOOD - General pattern
if transcription_text:
    let_llm_decide_operations_based_on_instructions()
```

### 2. **Never Force Specific Operations**

```python
# ❌ BAD - Assumes all audio questions need filter + sum
if has_audio and has_dataframe:
    tasks = [filter_task, sum_task]

# ✅ GOOD - LLM interprets instructions
tasks = generate_tasks_from_llm(transcription, artifacts)
```

### 3. **Never Make Assumptions About Question Structure**

```python
# ❌ BAD - Assumes specific workflow
if quiz_type == "demo-audio":
    download() -> transcribe() -> parse() -> filter() -> sum()

# ✅ GOOD - Adapts to any workflow
let_llm_plan_workflow_from_page_content()
```

---

## ✅ What TO Do

### 1. **Use LLM Decision-Making**

Let the LLM interpret instructions and decide operations:
- Transcription contains natural language instructions → LLM parses them
- Page has data sources → LLM determines what to fetch
- Artifacts exist → LLM decides what operations to perform

### 2. **Provide General Tools, Not Specific Workflows**

Build flexible tools that work for ANY question type:
- `dataframe_ops(op, params)` - works for filter, aggregate, transform, etc.
- `calculate_statistics(stats)` - works for sum, mean, median, count, etc.
- `transcribe_audio()` - works for any audio instructions
- `parse_csv()` - works for any CSV data

### 3. **Keep Prompts Generic**

```python
# ✅ GOOD - Generic prompt
"""
You have these artifacts: {artifacts}
You have these instructions: {transcription}
What operations are needed?
"""

# ❌ BAD - Demo-specific prompt  
"""
Filter the column where values >= cutoff, then sum them.
"""
```

### 4. **Teach the LLM, Don't Patch Around It**

```python
# ✅ GOOD - Educate the LLM about the system with clear examples
"""
PARAMETER NAMING CONVENTIONS:
- dataframe_ops: Use {"op": "sum", "params": {"dataframe_key": "df_0", "column": "sales"}}
  * The dataframe identifier goes in params.dataframe_key
- calculate_statistics: Use {"dataframe": "df_0", "stats": ["sum"], "columns": ["sales"]}
  * The dataframe identifier is a top-level parameter called 'dataframe'
"""

# ❌ BAD - Add code to accept any parameter variation
df_key = inputs.get("dataframe") or inputs.get("dataframe_key") or inputs.get("df")  # Patching!
```

**Why teaching is better:**
- LLM learns the correct pattern and applies it consistently
- Reduces need for code workarounds
- Makes the system more maintainable and testable
- Prompts serve as documentation and specification
- Catches mistakes early instead of silently working around them

**Teaching is mandatory:**
- Always update prompts/documentation FIRST when LLM makes mistakes
- Add explicit examples showing the correct pattern
- Use clear, unambiguous naming conventions
- Test if teaching works before adding fallback code

**When fallback code is acceptable (RARE):**
- Only for genuinely unpredictable LLM output variations that can't be taught
- Only after proving prompts can't solve it
- Must be generic (not quiz-specific)
- Must be documented as temporary workaround
- Should still log warnings when fallback is triggered

### 5. **Centralize All Tool Documentation**

**Single Source of Truth:** All tool-related information lives in `tool_definitions.py`

```python
# ✅ GOOD - Centralized in tool_definitions.py
def get_tool_usage_examples() -> str:
    return """
    AVAILABLE TOOLS:
    - fetch_from_api(url, method, headers, body): Call REST APIs
      * Returns: {"status_code": 200, "data": {...}}
      * Use dot notation for nested fields: "api_result.data.secret_code"
    - extract_patterns(text, pattern_type, custom_pattern): Extract patterns
      * pattern_type: "email", "url", "phone" (SINGULAR, not "emails")
    """

# All prompts import and use this:
from tool_definitions import get_tool_usage_examples
system_prompt = f"... {get_tool_usage_examples()} ..."
```

```python
# ❌ BAD - Duplicated documentation in multiple files
# llm_client.py
"""Tools: fetch_from_api, extract_patterns..."""

# task_generator.py  
"""Tools: fetch_from_api, extract_patterns..."""

# Different files = drift and inconsistency!
```

**Why centralization is critical:**
- **Single update point:** Change tool docs in ONE place
- **Automatic sync:** All prompts get updates immediately
- **No drift:** Can't have inconsistent documentation
- **Easier maintenance:** Find and update tool info quickly
- **Discoverability:** New tools visible everywhere automatically

**What to centralize:**
1. **Tool schemas** → `get_tool_definitions()` in `tool_definitions.py`
2. **Tool descriptions** → `get_tool_usage_examples()` in `tool_definitions.py`
3. **Usage patterns** → Include in `get_tool_usage_examples()`
4. **Parameter conventions** → Document in `get_tool_usage_examples()`
5. **Workflow examples** → Add to `get_tool_usage_examples()`

**Files that use centralized docs:**
- ✅ `llm_client.py` - Plan generation
- ✅ `task_generator.py` - Iterative task generation
- ✅ Any new prompts added in future

**When adding a new tool:**
1. Add schema to `get_tool_definitions()`
2. Add documentation to `get_tool_usage_examples()`
3. Implement in `tool_executors.py`
4. Route in `executor.py`
5. **Done!** All prompts automatically know about it

**See TOOL_MAINTENANCE.md for detailed guide**

### 5. **Validate Against Multiple Scenarios**

Test questions might have:
- Different audio instructions (not just "filter and sum")
- Different data formats (not just single-column CSV)
- Different operations (median, count, max, min, etc.)
- Different workflows (no audio, multiple files, etc.)
- Different final answer types (text, number, JSON, etc.)

### 6. **Keep Tool Documentation Synchronized**

**CRITICAL:** Tool definitions, implementations, and documentation must stay in sync

```python
# ✅ GOOD - Everything in sync
# tool_definitions.py
def get_tool_definitions():
    return [{"type": "function", "function": {"name": "extract_patterns", ...}}]

def get_tool_usage_examples():
    return """- extract_patterns(text, pattern_type): pattern_type = "email" (singular)"""

# tool_executors.py
def extract_patterns(text, pattern_type):
    if pattern_type == "email":  # Matches documentation
        ...

# executor.py
elif tool_name == "extract_patterns":
    result = extract_patterns(...)  # Routed correctly
```

```python
# ❌ BAD - Out of sync
# Documentation says: pattern_type = "emails" (plural)
# Code expects: pattern_type = "email" (singular)
# Result: LLM uses "emails", code rejects it ❌
```

**When documentation and code mismatch:**
1. ❌ **WRONG:** Add fallback code to accept both variations
2. ✅ **RIGHT:** Fix documentation to match code (or vice versa)
3. ✅ **TEACH:** Make parameter values explicit and unambiguous

---

## 🎯 Forcing vs. Intelligence

### When Forcing is ACCEPTABLE:

**Essential infrastructure only (preprocessing to make data available):**
```python
# ✅ OK - Ensures audio is transcribed
if has_audio_file and not has_transcription:
    force_transcribe_audio()

# ✅ OK - Ensures CSV is parsed when mentioned
if transcription_mentions_csv and not has_dataframe:
    force_parse_csv()

# ✅ OK - Ensures image/PDF is processed when keywords detected
if has_image and vision_keywords_in_text and not has_vision_result:
    force_vision_analysis()

# ✅ OK - Ensures API data is fetched when mentioned
if api_url_present and not has_api_response:
    force_api_fetch()
```

These are **enablers** - they make data available for the LLM to work with.

**Why these are acceptable:**
- They don't decide WHAT to do with the data
- They only ensure data is in a usable format
- LLM still interprets the data and decides operations
- Work for ANY question type (not demo-specific)

### When Forcing is WRONG:

**Analysis and calculations (LLM should decide these):**
```python
# ❌ WRONG - Assumes what calculations are needed
if has_dataframe and transcription_mentions_sum:
    force_calculate_sum()

# ❌ WRONG - Assumes specific filters
if "cutoff" in page_text:
    force_filter_operation()

# ❌ WRONG - Assumes operation sequence
if has_filtered_data:
    force_statistics_calculation()

# ❌ WRONG - Assumes answer format
if has_statistics:
    force_extract_sum_value()
```

These are **decisions** - they should be made by the LLM based on instructions.

**Why these are wrong:**
- They assume what the question is asking for
- They hardcode workflow sequences
- They remove LLM's ability to interpret instructions
- Won't work for variations of the question

---

## 🔍 How to Identify Overfitting

Ask these questions about any code change:

1. **"Will this work if the test question asks for MEDIAN instead of SUM?"**
   - If no → You're overfitting

2. **"Will this work if there's no audio, just text instructions?"**
   - If no → You're overfitting

3. **"Will this work if the CSV has 10 columns instead of 1?"**
   - If no → You're overfitting

4. **"Will this work if the filter condition is '<' instead of '>='?"**
   - If no → You're overfitting

5. **"Am I hardcoding based on what I see in demo questions?"**
   - If yes → You're overfitting

6. **"Will this work for images instead of CSV data?"** *(NEW - Multimodal test)*
   - If no → You're overfitting

7. **"Will this work if the question needs scraping instead of API calls?"** *(NEW - Data source test)*
   - If no → You're overfitting

8. **"Does this code assume a specific operation sequence?"** *(NEW - Workflow test)*
   - If yes → You're overfitting

9. **"Is this providing information or making decisions?"** *(NEW - Infrastructure vs Logic test)*
   - If decisions → You're overfitting

10. **"Would this still work if demo questions didn't exist?"** *(NEW - Ultimate test)*
    - If no → You're overfitting

11. **"Could this be fixed by improving the prompt instead?"** *(NEW - Prompt-first test)*
    - If yes → Improve the prompt first. Only add code if prompts provably can't solve it.

12. **"Am I patching around LLM mistakes instead of teaching it the right pattern?"** *(NEW - Education test)*
    - If patching → You're doing it wrong. Update prompts to teach the LLM the correct approach.

13. **"Does my prompt example match actual quiz text exactly?"** *(NEW - Example overfitting test)*
    - If yes → You're overfitting! Use generic examples only

14. **"Am I using words like MUST, REQUIRED, NEVER, NOT FOR in prompts?"** *(NEW - Forced logic test)*
    - If yes → You're forcing decisions, not teaching

15. **"Do my tool descriptions restrict rather than enable?"** *(NEW - Description test)*
    - If restricting → Remove negative constraints, describe capabilities only

16. **"Am I modifying test data to make the solver work?"** *(NEW - Test integrity test)*
    - If yes → You're hiding bugs! Fix the solver to handle real test conditions

17. **"Are all defined tools actually implemented?"** *(NEW - Implementation test)*
    - If no → Complete implementation before adding to tool definitions

18. **"Do similar tools have clear, distinct use cases in documentation?"** *(NEW - Tool clarity test)*
    - If no → Update docs to clarify when to use each tool

19. **"Are tool schemas, implementations, and documentation all in sync?"** *(NEW - Synchronization test)*
    - If no → Fix mismatches immediately. Documentation drift causes LLM errors

20. **"Am I duplicating tool documentation across multiple files?"** *(NEW - Centralization test)*
    - If yes → Move to centralized `get_tool_usage_examples()` in `tool_definitions.py`

---

## 📋 Checklist Before Committing Code

- [ ] Does this change add demo-specific logic? If yes, **remove it**
- [ ] Could this break on test questions with different patterns? If yes, **make it general**
- [ ] Am I forcing a specific operation the LLM should decide? If yes, **let LLM decide**
- [ ] Does this assume a particular question structure? If yes, **make it flexible**
- [ ] Have I tested this mentally against different scenarios? If no, **think through edge cases**
- [ ] **Could this be solved by improving the prompt instead? If yes, update prompts first and ONLY add code if prompts fail**
- [ ] **Am I adding fallback code to work around LLM mistakes? If yes, STOP and teach the LLM properly instead**
- [ ] **Does this make the system smarter or just paper over issues? Only commit if it teaches/enables, not patches**
- [ ] **Do any prompt examples match actual test questions? If yes, replace with generic examples**
- [ ] **Am I using MUST/REQUIRED/NOT in prompts to force tool selection? If yes, remove forced language**
- [ ] **Are tool descriptions positive (what they DO) vs negative (what they DON'T)? Must be positive**
- [ ] **Am I modifying test quizzes to fit the solver instead of fixing the solver? Test integrity is critical**
- [ ] **Are all tool definitions actually implemented and routed in executor? No orphaned definitions**
- [ ] **Do similar tools have clearly documented, distinct use cases? Avoid tool selection confusion**
- [ ] **Are tool schemas, implementations, and docs synchronized? Check tool_definitions.py vs tool_executors.py**
- [ ] **Is tool documentation centralized in get_tool_usage_examples()? No duplication across files**

---

## 🎓 Learning from Past Mistakes

### Mistake 1: Forced Filtering
**What we did:** Auto-filtered dataframes when transcription mentioned "greater than"
**Why it's wrong:** Test questions might need "less than", "equals", or no filter at all
**Fix:** Let LLM call `dataframe_ops` with appropriate condition

### Mistake 2: Forced Sum Calculation  
**What we did:** Auto-calculated sum when transcription mentioned "add"
**Why it's wrong:** Test questions might need mean, median, count, or other stats
**Fix:** Let LLM call `calculate_statistics` with appropriate stats array

### Mistake 3: Assumed Single-Column CSV
**What we did:** Used `columns[0]` to get the column name
**Why it's wrong:** Test questions might have multi-column CSVs, different target columns
**Fix:** Let LLM determine which column(s) to operate on

### Mistake 4: Patched Instead of Taught *(NEW)*
**What we did:** Added code to resolve artifact references when LLM used wrong names
**Why it's problematic:** Code patches over the issue instead of fixing the root cause
**Better fix:** 
1. **Primary**: Update prompts to teach LLM the correct referencing pattern
2. **Secondary**: Keep fallback code for edge cases, but prompt is the real solution
**Lesson:** Always try prompt improvements first, code second

### Mistake 5: Quiz-Specific Examples in Prompts *(NEW - CRITICAL)*
**What we did:** Added "build linear regression, predict y when x=50" to system prompt
**Why it's wrong:** This is the EXACT text from ml_challenge quiz - obvious overfitting!
**Why it seemed reasonable:** "We're teaching the LLM through examples"
**The trap:** Examples using actual quiz text are business logic, not teaching
**Fix:** Remove ALL quiz-specific examples. Use generic patterns only.
**Lesson:** If an example matches a specific test case exactly, it's overfitting

### Mistake 6: Forcing Tool Selection via Prompts *(NEW - CRITICAL)*
**What we did:** Added "You MUST use train_linear_regression (NOT call_llm, NOT dataframe_ops)"
**Why it's wrong:** This removes LLM's ability to choose appropriate tools
**Why it seemed reasonable:** "We're just clarifying which tool to use"
**The trap:** "MUST use X" and "NOT Y" are forced decisions, not teaching
**Fix:** Provide tool descriptions, let LLM choose based on task requirements
**Lesson:** Words like MUST, REQUIRED, NOT FOR are red flags for forced logic

### Mistake 7: Restrictive Tool Descriptions *(NEW - CRITICAL)*
**What we did:** Changed tool descriptions to say "NOT for machine learning"
**Why it's wrong:** Restricts LLM's intelligent tool selection
**Why it seemed reasonable:** "We're preventing misuse of tools"
**The trap:** Negative restrictions are business logic in disguise
**Fix:** Describe what each tool DOES, not what it DOESN'T do
**Lesson:** Tool descriptions should enable choices, not restrict them

### Mistake 8: Modifying Tests Instead of Solver *(NEW - CRITICAL)*
**What we did:** Added generic `/submit` endpoint to test quiz when LLM extracted wrong URL
**Why it's wrong:** Real test won't have this endpoint - we're fitting tests to solver
**Why it seemed reasonable:** "It makes the test work quickly"
**The trap:** Masks the real issue (LLM not extracting submit URL from page text)
**Fix:** 
1. Add submit URL to page text: "Submit your answer to: {url}"
2. Let LLM extract it from page content (as real test will require)
**Lesson:** **Never modify tests to fit the solver - modify the solver to handle real test conditions**

### Mistake 9: Missing Tool Implementation *(NEW - Infrastructure)*
**What we did:** Defined `transform_data` in tool schema but didn't implement it
**Why it's problematic:** LLM can't use tools that don't exist, falls back to wrong tools
**How we detected:** LLM chose `dataframe_ops` for pivot instead of `transform_data`
**Fix:**
1. Implement the tool in `tool_executors.py`
2. Add routing in `executor.py`
3. Update centralized documentation to clarify use cases
**Lesson:** Tool definitions, implementations, and documentation must stay in sync

### Mistake 10: Unclear Tool Boundaries *(NEW - Teaching)*
**What we did:** Had both `transform_data` and `dataframe_ops` without clear distinction
**Why it caused issues:** LLM didn't know when to use which tool for data operations
**Fix:** Updated centralized documentation with clear use cases:
- `transform_data`: RESTRUCTURING data shape (pivot, melt, transpose)
- `dataframe_ops`: ROW-LEVEL operations (filter, select, groupby)
- `calculate_statistics`: STATISTICAL analysis (sum, mean, median)
**Lesson:** Clear tool descriptions prevent tool selection confusion

### Mistake 11: Parameter Name Variations - The Wrong Approach ❌
**What we initially did:** LLM used different parameter names, so we added fallback code:
```python
# ❌ WRONG - Accepting any variation
df_key = inputs.get("dataframe") or inputs.get("dataframe_key") or inputs.get("df")
```
**Why this seemed reasonable:** "It's just infrastructure handling LLM variations"
**Why it's actually wrong:**
- Masks the real problem instead of fixing it
- LLM never learns the correct pattern
- Creates silent failures and hard-to-debug issues
- Makes the system less predictable
- Other parts of code expect specific parameter names

**The RIGHT approach - Teaching:**
```python
# ✅ CORRECT - Teach LLM explicit naming conventions
"""
PARAMETER NAMING CONVENTIONS:
- dataframe_ops: Use {"op": "...", "params": {"dataframe_key": "df_X", ...}}
- calculate_statistics: Use {"dataframe": "df_X", "stats": [...], ...}
"""
```

**Lesson:** **NEVER accept parameter variations as "acceptable infrastructure patching". Always teach the LLM the correct names through explicit documentation and examples. If LLM still makes mistakes, the prompts aren't clear enough.**

### Mistake 12: Documentation Mismatch with Implementation *(NEW - Synchronization)*
**What happened:** Documentation said `pattern_type: "emails"` but code expected `"email"` (singular)
**Why it's wrong:** LLM learns from documentation, uses "emails", code rejects it
**First instinct (WRONG):** Add fallback code to accept both variations
```python
# ❌ WRONG - Patching around the mismatch
pattern_type = inputs.get("pattern_type")
if pattern_type == "emails":
    pattern_type = "email"  # Convert plural to singular
```
**The RIGHT approach - Fix at source:**
```python
# ✅ CORRECT - Update documentation to match code
"""
- extract_patterns(text, pattern_type): Extract patterns from text
  * pattern_type: "email", "url", "phone" (SINGULAR, not "emails")
"""
```
**Why teaching is better:**
- Fixes root cause (misleading documentation)
- LLM learns correct parameter values
- No silent conversions hiding the problem
- Code stays clean and predictable

**Lesson:** **When documentation and code disagree, fix the documentation (or code) to match. Never add patching code to "handle both". Synchronization is critical.**

### Mistake 13: Duplicated Tool Documentation *(NEW - Centralization)*
**What we did:** Tool usage examples existed in multiple files:
- `llm_client.py` - Plan generation prompt
- `task_generator.py` - Task generation prompt
- Each with slightly different wording

**Why it's wrong:**
- Documentation drifts out of sync
- Adding/updating tools requires changes in multiple places
- Easy to forget a file and create inconsistencies
- LLM gets conflicting information from different prompts

**Fix:** Centralized all tool documentation:
```python
# tool_definitions.py - SINGLE SOURCE OF TRUTH
def get_tool_usage_examples() -> str:
    return """
    AVAILABLE TOOLS:
    - fetch_from_api(...): ...
    - extract_patterns(...): ...
    """

# llm_client.py - Uses centralized docs
from tool_definitions import get_tool_usage_examples
system_prompt = f"... {get_tool_usage_examples()} ..."

# task_generator.py - Uses same centralized docs
from tool_definitions import get_tool_usage_examples
prompt = f"... {get_tool_usage_examples()} ..."
```

**Benefits:**
- One place to update tool docs
- Impossible to have drift
- Adding tools updates all prompts automatically
- Easier to maintain and review

**Lesson:** **Centralize all tool-related information in `tool_definitions.py`. Never duplicate documentation. See TOOL_MAINTENANCE.md for the pattern.**

---

### Mistake 14: Over-Relying on Error Messages for Teaching *(NEW - CRITICAL)*
**What we did:** Added detailed error messages to catch LLM mistakes (wrong column names, invalid operators, etc.)
**Why it seemed reasonable:** "Error messages teach the LLM what went wrong"
**Why it's problematic:**
- Error-driven learning is reactive, not proactive
- Wastes LLM calls on failed attempts
- Slower execution (try → fail → retry)
- Error messages add complexity to implementation
- Still requires retry logic to recover

**The trap:** Thinking "good error messages = good teaching"

**Better approach - PROACTIVE TEACHING:**
```python
# ✅ EXCELLENT - Comprehensive upfront guidance
"""
═══════════════════════════════════════════════════════════════════════════
║               DATAFRAME COLUMN HANDLING - READ THIS FIRST                ║
═══════════════════════════════════════════════════════════════════════════

STEP 1: EXAMINE DATAFRAME METADATA
After parsing, you receive: {"columns": ["ID", "Name", "Value"], ...}

STEP 2: USE EXACT COLUMN NAMES (CASE-SENSITIVE)
- "Value" ≠ "value" ≠ "VALUE"
- Use names from 'columns' field EXACTLY

STEP 3: CHOOSE THE RIGHT OPERATION
A) FILTERING: {"op": "filter", "params": {"condition": "Value >= 100"}}
   - Operators: >= <= > < == !=
   - Format: "ColumnName operator value"
   
COMMON MISTAKES TO AVOID:
❌ Using lowercase when column is uppercase
❌ Using SQL-style: "WHERE Value > 100"
❌ Using unsupported operators: "is", "is not", "AND", "OR"

WORKFLOW PATTERN:
1. Parse data → Examine columns field
2. Choose operation based on task
3. Use EXACT column names from metadata
"""
```

**Benefits of proactive teaching:**
- LLM gets it right on FIRST attempt
- No wasted LLM calls on failures
- Faster execution (no retry cycles)
- Simpler implementation (less error handling)
- Better user experience (no error delays)

**When error messages are still valuable:**
- **Validation**: Catch genuinely unexpected edge cases
- **Debugging**: Help developers identify issues
- **Safety**: Prevent system crashes
- **But NOT for**: Teaching basic patterns LLM should know upfront

**The right balance:**
1. **Primary**: Comprehensive proactive teaching in tool definitions
2. **Secondary**: Clear validation errors for genuine edge cases
3. **Never**: Relying on error messages to teach common patterns

**Implementation pattern:**
```python
# ✅ GOOD - Proactive teaching prevents mistakes
"""
CRITICAL SECTIONS in tool_definitions.py:
- Visual emphasis (box drawing: ═══, ║)
- Step-by-step workflows
- Common mistakes sections with ❌ markers
- Decision trees for tool selection
- Explicit format specifications
- Inline examples showing correct usage
"""

# ✅ ACCEPTABLE - Validation for edge cases
if column not in df.columns:
    raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

# ❌ BAD - Using errors to teach common patterns
# If LLM commonly makes this mistake, teach it upfront instead!
if operator == 'is':
    raise ValueError("Use '==' instead of 'is'. Supported operators: >= <= > < == !=")
```

**Key metrics:**
- **Before (error-driven)**: 3-5 iterations, ~30-60s per quiz, complex error handling
- **After (proactive)**: 1-2 iterations, ~15-30s per quiz, simple validation only

**Lesson:** **Build robust, generalized prompts that teach patterns UPFRONT. Use visual emphasis, decision trees, and step-by-step workflows. Error messages are for validation, not primary teaching. If LLM commonly makes a mistake, improve the proactive teaching instead of relying on error feedback loops.**

---

### Mistake 15: Teaching Without Providing Data *(NEW - CRITICAL)*
**What we did:** Added extensive prompt instructions about using dataframe data, but didn't provide actual data
**Example:** 
```python
# Prompt said: "Transform ALL the data from {{df_0}}"
# But call_llm received: "Dataframe with 3 rows, 2 columns. Columns: ['name', 'value']"
# Result: LLM hallucinated fake data (Alice, Bob, Charlie) instead of transforming real data
```

**Why it's wrong:**
- Teaching what to do without providing means to do it
- Like asking someone to cook without giving ingredients
- LLM has no choice but to hallucinate or fail
- No amount of prompt improvement can fix missing data

**The trap:** "We documented the pattern clearly, why isn't it working?"

**Better approach - INFRASTRUCTURE INJECTION:**
```python
# ✅ EXCELLENT - Smart data injection in executor.py
def _prepare_llm_inputs(prompt: str, artifacts: dict) -> str:
    # Detect artifact references in prompt
    if '{{df_0}}' in prompt and 'df_0' in artifacts:
        artifact = artifacts['df_0']
        if 'dataframe_key' in artifact:
            # FETCH ACTUAL DATAFRAME
            df = dataframe_registry.get(artifact['dataframe_key'])
            # INJECT REAL DATA
            data_json = df.to_json(orient='records')
            enhanced_prompt = prompt.replace(
                'ARTIFACT METADATA',
                f"Dataframe with {len(df)} rows, {len(df.columns)} columns.\n"
                f"Columns: {list(df.columns)}\n"
                f"Data: {data_json}"
            )
```

**Benefits of smart injection:**
- LLM sees actual data, transforms correctly
- No hallucination possible
- Works for ANY dataframe, ANY transformation
- Teaching + data = success on first attempt

**Key distinction:**
- **Teaching alone**: "Here's how to use data" ❌ (Can't work without data)
- **Data alone**: Raw data dump ❌ (LLM doesn't know what to do)
- **Teaching + Smart Injection**: Instructions + actual data ✅ (Complete solution)

**Implementation pattern:**
```python
# In tool_definitions.py - TEACHING
"""
call_llm - Transform Data Pattern:
1. Reference dataframe: {{df_0}}
2. LLM receives the ACTUAL DATA
3. Transform and return in specified format
Example: "Convert columns: name→customer_id, value→amount"
"""

# In executor.py - INFRASTRUCTURE (lines 271-299)
if tool_name == "call_llm":
    # Detect dataframe artifacts
    for artifact_name, artifact_data in artifacts.items():
        if isinstance(artifact_data, dict) and 'dataframe_key' in artifact_data:
            # Fetch and inject actual data
            df = dataframe_registry.get(artifact_data['dataframe_key'])
            data_json = df.to_json(orient='records', date_format='iso')
            prompt = enhance_with_data(prompt, artifact_name, df, data_json)
```

**Real-world impact:**
- **Before injection**: LLM hallucinated "Alice, Bob, Charlie" ❌
- **After injection**: LLM transformed actual "Alpha, Gamma, Beta" ✅
- **Iterations**: 1 (got it right immediately)

**When injection is needed:**
- ✅ Data transformations (renaming columns, format changes)
- ✅ Data analysis requiring actual values
- ✅ Calculations that need real numbers
- ✅ Pattern detection in actual data

**When injection is NOT needed:**
- ✅ Metadata operations (column names, data types)
- ✅ Planning (deciding which operations to perform)
- ✅ Simple extractions (getting values from JSON/text)

**Lesson:** **Teaching LLM patterns is essential, but if the task requires actual data, infrastructure must inject it. Prompts teach WHAT to do, injection provides MEANS to do it. Both are required for data transformation tasks. No amount of teaching can substitute for missing data.**

---

## 🚀 Performance Optimizations (That Don't Violate Principles)

### Optimization 1: Fast Completion Checks ✅
**What we did:** Detect obvious completion states without LLM calls
```python
# Check for final results: statistics, charts, vision results, API values
if 'statistics' in artifact or 'chart_path' in artifact:
    return {"complete": True}
```
**Why it's acceptable:** Generic pattern detection (ANY statistics, ANY chart), not demo-specific
**Impact:** Saves 2-4 LLM calls per quiz, ~10-20 seconds

### Optimization 2: Operation History Tracking ✅
**What we did:** Show LLM what operations were completed
```python
artifacts['_COMPLETED_OPERATIONS'] = [
    "✓ Filtered dataframe -> df_1",
    "✓ Calculated statistics"
]
```
**Why it's acceptable:** Provides visibility, doesn't force next action. LLM still decides workflow
**Impact:** Prevents duplicate operations, saves ~10-20 seconds

### Optimization 3: Latest Artifact References ✅
**What we did:** Track latest dataframe/image/API response
```python
artifacts['_LATEST_DATAFRAME'] = 'df_2'
artifacts['_LATEST_IMAGE'] = '/path/to/image.png'
```
**Why it's acceptable:** Navigation aid only. LLM chooses which artifact to use
**Impact:** Helps LLM find correct resources in multi-step workflows

### Optimization 4: Conservative JSON Cleanup ✅
**What we did:** Remove O1 model reasoning contamination from function arguments
```python
# Only clean if contamination patterns detected
if any(p in value for p in ['}}', '//Oops', '\nJk']):
    apply_aggressive_cleanup()
```
**Why it's acceptable:** Infrastructure-level sanitization, not business logic
**Impact:** Prevents JSON parse errors while handling model quirks

**Key Principle**: Optimizations should be **infrastructure improvements** (caching, cleanup, fast paths), not **decision-making logic** (hardcoded workflows, forced operations)

---

## 💡 The Golden Rule

> **If you're writing code that only works for the specific demo questions you can see,  
> you're doing it wrong.**

The system should be intelligent enough to handle:
- Questions we haven't seen yet
- Instructions we haven't tested
- Data structures we haven't encountered
- Operations we haven't anticipated

**Trust the LLM to interpret and decide. Don't hardcode the path.**

**Additional Golden Rules:**

> **If your prompt example matches actual test question text,  
> you're overfitting through prompts instead of code.**

> **If you're telling the LLM what NOT to do instead of teaching capabilities,  
> you're forcing decisions instead of enabling intelligence.**

> **If you use MUST/REQUIRED/NEVER in prompts to force tool selection,  
> you've replaced LLM decision-making with hardcoded business logic.**

---

## 🚀 Success Criteria

The system is correctly designed when:

✅ We can add a completely new question type without changing code  
✅ The LLM drives the workflow based on instructions  
✅ Tools are generic and composable  
✅ No hardcoded assumptions about data or operations  
✅ Works on test questions we've never seen  

---

## 📝 Code Review Questions

When reviewing any change, ask:

1. "Is this adding intelligence or hardcoding a pattern?"
2. "Will future questions with different patterns break this?"
3. "Are we trusting the LLM or replacing it with hardcoded logic?"
4. "Is this a general tool or a demo-specific workaround?"
5. **"Is this infrastructure (preprocessing, caching, cleanup) or business logic (operations, calculations)?"** *(NEW)*
6. **"Does this work for ALL multimodal data types (data, images, PDFs, APIs, audio, scraping)?"** *(NEW)*
7. **"Am I using generic keywords/patterns or demo-specific values?"** *(NEW)*
8. **"Have I tried fixing this in prompts before adding code?"** *(NEW - Prompt-first principle)*
9. **"Is this teaching the LLM or working around it?"** *(NEW - Education vs Patching)*
10. **"Am I modifying test data/infrastructure to make solver work?"** *(NEW - Test integrity)*
11. **"Are tool definitions, implementations, and docs all in sync?"** *(NEW - Tool completeness)*

**If the answer to any question reveals overfitting, refactor to be general.**

**Golden workflow: Prompt improvement → Test → Code fallback only if needed**

---

## 🎨 Infrastructure vs Business Logic

### ✅ Infrastructure (ACCEPTABLE)
**Characteristics:**
- Makes data available in usable format
- Provides context/visibility to LLM
- Optimizes performance without changing decisions
- Uses generic patterns, not specific values

**Examples:**
```python
✅ force_transcribe_audio()      # Makes audio content available
✅ force_parse_csv()              # Makes CSV data available
✅ track_completed_operations()   # Shows LLM what's done
✅ fast_completion_checks()       # Skips LLM when obvious
✅ clean_json_contamination()     # Fixes unpredictable model quirks
✅ add_submit_url_to_page_text()  # Makes data extractable (infrastructure, not business logic)
```

**Note:** Even infrastructure code should be minimized. Ask: "Can prompts handle this instead?"

**REMOVED Examples (these are NOT acceptable):**
```python
❌ resolve_artifact_references()  # Should teach LLM correct references
❌ handle_parameter_variations()  # Should teach LLM correct parameter names
❌ accept_any_dataframe_param()   # Should document exact parameter names
```
**Why removed:** These patch around LLM mistakes instead of teaching proper patterns.

### ❌ Business Logic (OVERFITTING)
**Characteristics:**
- Decides what operations to perform
- Assumes what answer is needed
- Hardcodes workflow sequences
- Uses demo-specific values

**Examples:**
```python
❌ if "cutoff" in page: filter_column()        # Assumes specific operation
❌ if has_filter: calculate_sum()              # Assumes workflow sequence
❌ if transcription_has_add: force_sum()       # Assumes what "add" means
❌ value = columns[0]                          # Assumes single column
❌ add_generic_submit_endpoint()               # Modifies test to fit solver (hides bugs)
❌ define_tool_without_implementing()          # Creates orphaned tool definitions
```

**The Line:**
- Infrastructure: "Here's the data/context, LLM decides what to do"
- Business Logic: "I know what to do, forcing specific operations"

---

## 📊 Dataframe Referencing Convention

### The Problem
When tools parse data files (CSV, Excel, JSON), they store dataframes in a registry with keys like `df_0`, `df_1`, etc. The parse tool returns an artifact with metadata:

```json
{
  "dataframe_key": "df_2",
  "shape": (50, 2),
  "columns": ["x", "y"],
  "sample": {...}
}
```

**Critical Issue**: The LLM needs to reference this dataframe in subsequent operations, but there are TWO different identifiers:
1. **Artifact name** (what the LLM calls it): `"df_0"`, `"regression_data"`, etc.
2. **Registry key** (where it's actually stored): `"df_2"`, `"df_3"`, etc.

### ❌ Wrong Approaches

**Wrong 1: Using artifact name directly**
```python
# LLM generates this:
{"dataframe_key": "df_0"}  # "df_0" is the artifact name, not the registry key

# Result: ERROR - "df_0" not found in registry (actual key is "df_2")
```

**Wrong 2: Teaching LLM to use nested template variables**
```python
# Teaching LLM to do this:
{"dataframe_key": "{{df_0.dataframe_key}}"}

# Problem: Executor doesn't support nested field access in templates
# It strips {{}} and looks for artifact named "df_0.dataframe_key"
```

**Wrong 3: Patching with parameter variations**
```python
# Adding code to accept any parameter name:
df_key = inputs.get("dataframe") or inputs.get("dataframe_key") or inputs.get("df")

# Problem: Masks the issue, LLM never learns correct pattern
```

### ✅ Correct Approach: Infrastructure Resolution

**Convention**: LLM uses artifact names in tool calls, executor resolves them to registry keys

**1. LLM generates plan with artifact references:**
```python
{
  "tool_name": "train_linear_regression",
  "inputs": {
    "dataframe_key": "{{df_0}}",  # Template marker indicates resolution needed
    "feature_columns": ["x"],
    "target_column": "y"
  }
}
```

**2. Executor resolves artifact → registry key:**
```python
# In executor.py - GENERIC INFRASTRUCTURE
if input_value == "{{df_0}}":
    artifact = artifacts["df_0"]  # Get artifact metadata
    if "dataframe_key" in artifact:
        resolved_value = artifact["dataframe_key"]  # Extract "df_2"
```

**3. Tool receives resolved value:**
```python
train_linear_regression(dataframe_key="df_2", ...)  # Actual registry key
```

### Implementation Locations

**Teaching (tool_definitions.py):**
```python
"""
DATAFRAME REFERENCING:
- Parse tools return: {"dataframe_key": "df_X", "columns": [...], ...}
- To use the dataframe, reference the ARTIFACT NAME with template markers
- Example: parse_csv produces artifact "regression_data"
  → Use "dataframe_key": "{{regression_data}}"
  → Executor resolves to actual registry key "df_2"

CORRECT:
  {"dataframe_key": "{{df_0}}", ...}  ✅ Template marker → executor resolves
  
WRONG:
  {"dataframe_key": "df_0", ...}      ❌ Literal string → not found in registry
  {"dataframe_key": "df_2", ...}      ❌ Hardcoded key → won't work for other quizzes
"""
```

**Resolution (executor.py lines 140-174):**
```python
# GENERIC INFRASTRUCTURE: Resolve {{artifact}} references in ALL tool inputs
for input_key, input_value in inputs.items():
    if isinstance(input_value, str) and '{{' in input_value:
        artifact_ref = input_value.strip('{}').strip()
        if artifact_ref in artifacts:
            artifact_data = artifacts[artifact_ref]
            if isinstance(artifact_data, dict) and 'dataframe_key' in artifact_data:
                resolved_inputs[input_key] = artifact_data['dataframe_key']
```

### Why This Is Acceptable Infrastructure

**Passes all design principle tests:**

✅ **Generic pattern**: Works for ANY artifact with dataframe_key field  
✅ **Not quiz-specific**: Doesn't assume specific quiz structure  
✅ **Enables, doesn't decide**: LLM still chooses which artifact to use  
✅ **Makes data available**: Resolves references so tools can access dataframes  
✅ **No business logic**: Doesn't decide what operations to perform  
✅ **Teachable**: LLM can learn the `{{artifact}}` pattern  

**What it does NOT do:**
- ❌ Decide which dataframe to use (LLM chooses)
- ❌ Decide what operations to perform (LLM chooses)
- ❌ Accept parameter name variations (exact names required)
- ❌ Hardcode quiz-specific values (works for any artifact name)

### Different Tools, Different Parameters

**Important**: Different tools use different parameter names for dataframe references. The executor resolves `{{artifact}}` for ALL of them, but the parameter names themselves are exact:

```python
# dataframe_ops - nested in params
{"op": "filter", "params": {"dataframe_key": "{{df_0}}", "condition": "x > 50"}}

# calculate_statistics - top-level "dataframe" 
{"dataframe": "{{df_0}}", "stats": ["mean"], "columns": ["x"]}

# train_linear_regression - top-level "dataframe_key"
{"dataframe_key": "{{df_0}}", "feature_columns": ["x"], "target_column": "y"}
```

**Teaching emphasizes**: Use the EXACT parameter name for each tool, but use `{{artifact}}` for the value.

### Checklist

When dealing with dataframe references:

- [ ] Does teaching show `{{artifact}}` pattern for all dataframe tools?
- [ ] Does executor have generic resolution for `{{artifact}}` → registry key?
- [ ] Are parameter names exact (not accepting variations)?
- [ ] Is resolution generic (works for any artifact name)?
- [ ] Does this enable LLM choice (not force specific operations)?
