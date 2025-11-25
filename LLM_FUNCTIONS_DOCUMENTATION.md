# LLM Functions Documentation

## Overview
This document catalogs all LLM-calling functions in the quiz solver, their purposes, prompts, and when they're used.

---

## 1. `call_llm()` - Central LLM Function

**Location:** `llm_client.py` lines 23-96

**Purpose:** Single centralized function for ALL LLM interactions

**Signature:**
```python
async def call_llm(
    prompt: str, 
    system_prompt: str = None, 
    max_tokens: int = 2000, 
    temperature: float = 0,
    use_tools: bool = False,
    tool_choice: str = "auto"
) -> Any
```

**Modes:**
- **Text mode** (`use_tools=False`): Returns string response - used for JSON generation, text analysis
- **Tool mode** (`use_tools=True`): Returns message object with tool_calls - used for function calling

**Used by:**
- `call_llm_for_plan()` - text mode for JSON plan generation
- `call_llm_with_tools()` - tool mode for function calling (currently unused)
- Tool executors that need LLM processing

**Key features:**
- Handles authentication with SECRET
- Routes to OpenRouter API
- Optionally includes tool definitions for function calling
- Consistent logging with `[LLM_CALL]` prefix

---

## 2. `call_llm_for_plan()` - Plan Generation (TEXT-BASED)

**Location:** `llm_client.py` lines 389-530

**Purpose:** Generate execution plan as JSON text (NOT using function calling)

**Signature:**
```python
async def call_llm_for_plan(page_data: Dict[str, Any], previous_attempts: List[Any] = None) -> str
```

**Returns:** String containing JSON plan

**System Prompt Structure:**
```
You are an execution planner for an automated quiz-solving agent.

OUTPUT FORMAT - YOU MUST OUTPUT VALID JSON ONLY:
{
  "submit_url": "...",
  "tasks": [...],
  "final_answer_spec": {...},
  "request_body": {...}
}

WHEN NO TOOLS ARE NEEDED:
- Set tasks: []
- Set final_answer_spec.from to THE ACTUAL ANSWER VALUE

WHEN TOOLS ARE NEEDED:
- Create task objects
- Set final_answer_spec.from to reference artifact key

{get_tool_usage_examples()}  <-- Tool documentation inserted here

CRITICAL ARTIFACT REFERENCE RULES:
- parse_csv creates {"dataframe_key": "df_0"}
- Use actual keys in subsequent tasks
...
```

**User Prompt Structure:**
```
QUIZ PAGE DATA:
Text: {page_data['text']}
Code blocks: {page_data['code_blocks']}
Links: {page_data['links']}
Audio sources: {audio_sources}
Image sources: {image_sources}
HTML preview: {html[:500]}...

ANALYZE THIS QUIZ AND GENERATE THE EXECUTION PLAN JSON.

Requirements:
1. Extract submit URL
2. Determine required operations
3. Structure as valid JSON
...
```

**Called by:** `executor.py` line 915 during quiz execution

**Current issue:** LLM generates plans with old `transform_data` structure instead of new `dataframe_ops` structure

---

## 3. `call_llm_with_tools()` - Function Calling Mode (UNUSED)

**Location:** `llm_client.py` lines 97-388

**Purpose:** Generate execution plan using OpenAI function calling (NOT currently used)

**Signature:**
```python
async def call_llm_with_tools(page_data: Dict[str, Any], previous_attempts: List[Any] = None) -> Dict[str, Any]
```

**Returns:** Dictionary with parsed function calls

**Status:** ⚠️ **NOT CURRENTLY USED** - Replaced by text-based `call_llm_for_plan()`

**Why not used:** 
- Text-based JSON generation is simpler
- More flexible for complex plan structures
- Easier to debug

**If we switched to this:**
- Would send actual tool schemas to LLM
- LLM would return structured function calls
- Might eliminate parameter name issues
- More complex to convert to task format

---

## Function Call Flow

### Quiz Solving Flow:
```
executor.py (line 915)
  ↓
call_llm_for_plan(page_data, previous_attempts)
  ↓
call_llm(prompt, system_prompt, use_tools=False)  <-- TEXT MODE
  ↓
Returns: JSON string with plan
  ↓
executor.py parses JSON, executes tasks
```

### Why LLM isn't learning new structure:
1. **Tool documentation** (`get_tool_usage_examples()`) is text-based
2. **LLM is interpreting text**, not using formal schemas
3. **No schema validation** - LLM can generate any JSON structure
4. **Documentation may be unclear** - needs more explicit examples

---

## Documentation Sources

The LLM learns about tools from:

1. **`get_tool_usage_examples()`** in `tool_definitions.py` (lines 19-82)
   - Text-based tool descriptions
   - Usage patterns
   - Critical patterns
   - Parameter naming conventions

2. **System prompt examples** in `call_llm_for_plan()`
   - JSON output format
   - Artifact reference rules
   - When to use tools vs direct answers

3. **Tool schemas** in `get_tool_definitions()` (lines 88-807)
   - ONLY used if `use_tools=True` (function calling mode)
   - NOT sent to LLM in current text-based planning

---

## Current Problem Analysis

### What's happening:
```
LLM generating: {"operation": "pivot", "dataframe": "df_0", ...}
System expects: {"op": "pivot", "params": {"dataframe_key": "df_0", ...}}
```

### Why:
1. LLM was previously trained on `transform_data` tool structure
2. We removed `transform_data` from schemas
3. We updated text documentation
4. But LLM still uses old pattern (cache? unclear docs?)

### Solutions tried:
1. ✅ Removed `transform_data` tool completely
2. ✅ Added pivot/melt/transpose to `dataframe_ops`
3. ✅ Updated documentation with explicit examples
4. ✅ Added CRITICAL notes showing correct structure
5. ❌ Still failing - LLM not reading/understanding docs

### Next steps to try:
1. Add explicit JSON example in system prompt (not just tool docs)
2. Show exact structure in OUTPUT FORMAT section
3. Consider switching to function calling mode (`use_tools=True`)
4. Add validation/correction layer before execution

---

## Tool Documentation Location

**File:** `tool_definitions.py`

**Function:** `get_tool_usage_examples()` (lines 19-82)

**Content:**
- PARAMETER NAMING CONVENTIONS (lines 22-31)
- CRITICAL PATTERNS (lines 33-41)
- Data Transformation section (lines 43-50)
- All other tool categories

**This is what LLM sees** when generating plans in text mode.

**Current content for dataframe_ops:**
```
Data Transformation:
- dataframe_ops(op, params): All DataFrame operations including transformations
  * CRITICAL STRUCTURE: {"op": "pivot", "params": {"dataframe_key": "df_0", "index": "...", ...}}
  * NOT this: {"operation": "pivot", "dataframe": "df_0", ...} ❌
  * Pivot example: {"op": "pivot", "params": {"dataframe_key": "df_0", ...}}
```

---

## Recommendation

**Add explicit JSON example to system prompt OUTPUT FORMAT section:**

Instead of just showing generic task structure, show SPECIFIC dataframe_ops example:

```json
{
  "tasks": [
    {
      "id": "task_1",
      "tool_name": "parse_csv",
      "inputs": {"url": "http://example.com/data.csv"}
    },
    {
      "id": "task_2", 
      "tool_name": "dataframe_ops",
      "inputs": {
        "op": "pivot",
        "params": {
          "dataframe_key": "df_0",
          "index": "category",
          "columns": "month",
          "values": "sales"
        }
      }
    }
  ]
}
```

This puts the correct structure RIGHT in front of the LLM during plan generation.
