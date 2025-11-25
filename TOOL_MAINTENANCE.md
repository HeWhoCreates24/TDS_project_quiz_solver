# Tool Maintenance Guide

## Problem Solved
Previously, when adding a new tool, you had to manually update prompts in multiple files:
- ❌ `llm_client.py` - `call_llm_for_plan()` prompt
- ❌ `llm_client.py` - `call_llm_with_tools()` prompt  
- ❌ `task_generator.py` - `generate_next_tasks()` prompt
- ❌ Risk of inconsistency and missing updates

## Solution: Centralized Tool Documentation

### Single Source of Truth
All tool documentation is now in `tool_definitions.py`:

```python
def get_tool_usage_examples() -> str:
    """
    Generate tool usage examples dynamically.
    This ensures prompts stay in sync with available tools.
    """
    return """
    AVAILABLE TOOLS AND USAGE:
    
    Web & Data Fetching:
    - render_js_page(url): ...
    - download_file(url): ...
    
    Multimedia:
    - transcribe_audio(audio_path): ...
    - analyze_image(image_path, task): Vision AI for OCR...
    ...
    """
```

### Usage in Prompts
All prompts now import and use this function:

```python
from tool_definitions import get_tool_definitions, get_tool_usage_examples

# In prompts:
system_prompt = f"""
...
{get_tool_usage_examples()}
...
"""
```

## Adding a New Tool

### ✅ CORRECT Process (3 steps):

1. **Add tool definition** in `tool_definitions.py`:
   ```python
   def get_tool_definitions():
       return [
           ...
           {
               "type": "function",
               "function": {
                   "name": "new_tool",
                   "description": "Clear description",
                   "parameters": {...}
               }
           }
       ]
   ```

2. **Update centralized documentation** in `get_tool_usage_examples()`:
   ```python
   def get_tool_usage_examples():
       return """
       ...
       New Category:
       - new_tool(param1, param2): What it does and when to use it
         * Usage notes
         * Critical patterns
       ...
       """
   ```

3. **Implement the tool** in `tool_executors.py` and hook it up in `executor.py`

### ❌ WRONG Process (old way):
- Manually updating 3+ different prompt strings
- Risk of forgetting a file
- Inconsistent documentation
- Maintenance nightmare

## Files Using Centralized Documentation

✅ `llm_client.py`:
- `call_llm_for_plan()` - Initial plan generation
- Uses `get_tool_usage_examples()` to show available tools

✅ `task_generator.py`:
- `generate_next_tasks()` - Iterative task generation  
- Uses `get_tool_usage_examples()` when instructions present

## Benefits

1. **Single Update Point**: Add tool documentation in ONE place
2. **Automatic Sync**: All prompts get the update automatically
3. **Consistency**: No more drift between different prompts
4. **Maintainability**: Easy to find and update tool docs
5. **Discoverability**: New tools are immediately visible to LLM

## Example: Adding Vision Tool

### Before (Manual Updates Needed):
```python
# llm_client.py call_llm_for_plan
- analyze_image(...): ...  # ← Add here

# llm_client.py call_llm_with_tools  
- analyze_image(...): ...  # ← Add here too

# task_generator.py
- analyze_image(...): ...  # ← And here!
```

### After (Single Update):
```python
# tool_definitions.py ONLY
def get_tool_usage_examples():
    return """
    ...
    Multimedia:
    - analyze_image(image_path, task): Vision AI for OCR, description, etc.
      * task="ocr" extracts text/numbers from images
      * Use for ANY image analysis, NOT call_llm
    ...
    """
```

All prompts automatically get the update! ✅

## Design Principles Compliance

This pattern follows DESIGN_PRINCIPLES.md:
- ✅ **Infrastructure**: Tool docs are infrastructure, not business logic
- ✅ **Generic**: Works for ANY tool type
- ✅ **Maintainable**: Single source of truth
- ✅ **No Overfitting**: Tool descriptions are general-purpose
- ✅ **Prompt-First**: Easy to improve tool descriptions

## Verification

To verify all prompts use centralized docs:
```bash
# Should find imports in these files:
grep "get_tool_usage_examples" quiz_solver/llm_client.py
grep "get_tool_usage_examples" quiz_solver/task_generator.py

# Should find usage in prompts:
grep -A5 "get_tool_usage_examples()" quiz_solver/llm_client.py
grep -A5 "get_tool_usage_examples()" quiz_solver/task_generator.py
```

## Future Additions

When adding new tools, remember:
1. Tool definition → `get_tool_definitions()`
2. Tool documentation → `get_tool_usage_examples()`  
3. Tool implementation → `tool_executors.py`
4. Tool routing → `executor.py`

**That's it!** No need to hunt through prompts. ✅
