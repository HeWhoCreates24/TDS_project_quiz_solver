"""LLM client functions for making API calls to language models
Handles plan generation, tool calling, and general LLM interactions
"""
import os
import re
import json
import logging
from typing import Any, Dict, List
import httpx
from tool_definitions import get_tool_definitions

logger = logging.getLogger(__name__)

# Use a non-reasoning model for reliable JSON output
# Reasoning models (gpt-5-nano) consume tokens on reasoning, leaving no content
model = "openai/gpt-4o-mini"  # Fast, reliable, and cost-effective


async def call_llm(prompt: str, system_prompt: str = None, max_tokens: int = 2000, temperature: float = 0) -> str:
    """Call LLM with given prompt"""
    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant."
    
    OPEN_AI_BASE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
    API_KEY = os.getenv("API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    json_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    logger.info(f"[LLM_CALL] System prompt: {system_prompt[:100]}...")
    logger.info(f"[LLM_CALL] User prompt: {prompt[:200]}...")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(OPEN_AI_BASE_URL, headers=headers, json=json_data, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer_text = data["choices"][0]["message"]["content"]
        logger.info(f"[LLM_RESPONSE] Content: {answer_text[:300]}...")
        return answer_text.strip()


async def call_llm_with_tools(page_data: Dict[str, Any], previous_attempts: List[Any] = None) -> Dict[str, Any]:
    """
    Generate execution plan using OpenAI function/tool calling API.
    This is more robust than prompt-based JSON generation.
    """
    # Build context from previous attempts
    previous_context = ""
    if previous_attempts and len(previous_attempts) > 0:
        previous_context = "\n\nPREVIOUS ATTEMPTS:\n"
        for i, attempt in enumerate(previous_attempts[-3:], 1):
            previous_context += f"\nAttempt {i}:\n"
            previous_context += f"Answer submitted: {attempt.answer}\n"
            previous_context += f"Result: {attempt.result}\n"
    
    system_prompt = """You are an execution planner for an automated quiz-solving agent.
Analyze the quiz page and USE FUNCTION CALLING to indicate which tools to use.

CRITICAL RULES:
1. ONLY call tools to GET information needed for the answer
2. DO NOT call tools to submit the answer (we handle submission separately)
3. If the answer is already in the page or is obvious, respond directly WITHOUT calling tools
4. For data analysis tasks, call MULTIPLE tools in sequence (e.g., parse_csv → calculate_statistics)

Common patterns:
- Page says "answer anything" or "any value" → NO TOOLS, respond with a simple answer like "anything you want"
- Page has links to visit/scrape → CALL render_js_page(url) to get content from those pages
- Page references CSV/data and asks for sum/stats → CALL parse_csv(url) AND calculate_statistics or dataframe_ops
- Page has complex text → CALL call_llm(prompt) to extract information

NEVER call fetch_from_api or any tool to POST to /submit - we handle submissions automatically.

DATA ANALYSIS EXAMPLES:
✅ "Sum numbers in data.csv" → CALL parse_csv, THEN calculate_statistics with stats=["sum"]
✅ "Filter values > 1000" → CALL parse_csv, THEN dataframe_ops with op="filter"
✅ "Count rows where X > Y" → CALL parse_csv, THEN dataframe_ops with op="filter", THEN count

SCRAPING EXAMPLES:
✅ Page: "Visit https://example.com/secret" → CALL render_js_page({"url": "https://example.com/secret"})
✅ Page: "The answer can be anything" → NO TOOLS, respond: "Hello World"
❌ WRONG: Calling fetch_from_api to POST to /submit (we do this for you)"""

    # Check if page references data files and multimedia
    page_text_lower = page_data['text'].lower()
    has_csv_link = any('.csv' in str(link).lower() for link in page_data.get('links', []))
    has_audio = len(page_data.get('audio_sources', [])) > 0
    has_video = len(page_data.get('video_sources', [])) > 0
    has_images = len(page_data.get('image_sources', [])) > 0
    
    data_hint = ""
    tool_choice_mode = "auto"
    
    # Check for multimedia first - audio often contains instructions
    if has_audio:
        audio_url = page_data['audio_sources'][0]
        logger.info(f"[AUDIO_DETECTED] Page has audio source: {audio_url}")
        logger.info(f"[PAGE_TEXT] Full text: {page_data['text']}")
        logger.info(f"[MULTIMEDIA] Audio sources: {page_data.get('audio_sources', [])}")
        
        tool_choice_mode = "required"
        
        csv_link = None
        if has_csv_link:
            csv_link = next((link for link in page_data['links'] if '.csv' in str(link).lower()), None)
        
        data_hint = f"""

🚨 CRITICAL INSTRUCTION - AUDIO + DATA ANALYSIS TASK 🚨

The page contains an AUDIO file: {audio_url}
{f'The page also contains a CSV file: {csv_link}' if csv_link else ''}

⚠️ THE AUDIO FILE CONTAINS SPOKEN INSTRUCTIONS FOR WHAT TO DO WITH THE DATA ⚠️

You MUST call ALL of these tools in your SINGLE response (we will execute them in sequence):

1. download_file
   Arguments: {{"url": "{audio_url}"}}
   This downloads the audio file and returns {{"path": "...", ...}}

2. transcribe_audio
   Arguments: {{"audio_path": "${{download_file_result_1.path}}"}}
   Use the path from step 1 to transcribe the audio to text

{f'''3. parse_csv
   Arguments: {{"url": "{csv_link}"}}
   Load the CSV data into a dataframe

4. Based on what the transcribed audio says, call the appropriate analysis tool:
   - If audio says "sum": calculate_statistics with {{"dataframe": "df_0", "stats": ["sum"]}}
   - If audio says "mean/average": calculate_statistics with {{"dataframe": "df_0", "stats": ["mean"]}}
   - If audio says "count": calculate_statistics with {{"dataframe": "df_0", "stats": ["count"]}}
   - If audio says "max/maximum": calculate_statistics with {{"dataframe": "df_0", "stats": ["max"]}}
   - If audio says "min/minimum": calculate_statistics with {{"dataframe": "df_0", "stats": ["min"]}}''' if csv_link else ''}

IMPORTANT: Call ALL required tools in ONE response. Don't call just download_file - call download_file AND transcribe_audio{' AND parse_csv AND the analysis tool' if csv_link else ''}.
The answer is NOT the file metadata - it's the result from analyzing the data based on audio instructions.
"""
    elif has_csv_link:
        csv_link = next((link for link in page_data['links'] if '.csv' in str(link).lower()), None)
        logger.info(f"[CSV_DETECTED] Page has CSV link: {csv_link}")
        logger.info(f"[PAGE_TEXT] Full text: {page_data['text']}")
        
        # Only force CSV analysis if page explicitly asks for it
        analysis_keywords = ['sum', 'total', 'count', 'average', 'mean', 'filter', 'calculate', 'add up', 'compute']
        needs_analysis = any(kw in page_text_lower for kw in analysis_keywords)
        
        if needs_analysis:
            tool_choice_mode = "required"
            data_hint = f"""

🚨 CSV DATA ANALYSIS TASK 🚨

The page contains a CSV file: {csv_link}
Call parse_csv to load it, then perform the requested analysis.
"""
    
    prompt = f"""
QUIZ PAGE DATA:
Text: {page_data['text']}
Code blocks: {page_data['code_blocks']}
Links: {page_data['links']}
HTML preview: {page_data['html'][:500]}...{previous_context}{data_hint}

TASK: Determine ALL tools needed to GET the complete answer.

IMPORTANT: If the quiz requires data analysis (sum, filter, count, etc.), you must call:
1. First tool to load data (parse_csv, parse_excel, etc.)
2. Second tool to analyze data (calculate_statistics, dataframe_ops, etc.)

DECISION TREE:
1. Does the page say the answer can be "anything" or "any value"? → NO TOOLS, just respond with any string
2. Does the page ask to analyze/sum/filter data AND there's a CSV/data file? → CALL parse_csv + calculate_statistics
3. Does the page have links you need to visit (HTML pages)? → CALL render_js_page for each link
4. Is the answer obvious from the page text? → NO TOOLS, respond with the answer

Remember: Call ALL tools needed in ONE response. We can execute multiple tools in sequence.
"""

    OPEN_AI_BASE_URL = os.getenv("LLM_BASE_URL", "https://aipipe.org/openrouter/v1/chat/completions")
    API_KEY = os.getenv("API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Get tool definitions
    tools = get_tool_definitions()
    
    json_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "tools": tools,
        "tool_choice": tool_choice_mode,
        "temperature": 0,
        "max_tokens": 3000,
    }
    
    if tool_choice_mode == "required":
        logger.info(f"[LLM_TOOLS] Forcing tool usage (tool_choice=required) due to CSV detection")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(OPEN_AI_BASE_URL, headers=headers, json=json_data, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        message = data["choices"][0]["message"]
        
        # Check if model wants to call tools
        if message.get("tool_calls"):
            logger.info(f"[LLM_TOOLS] Model requested {len(message['tool_calls'])} tool call(s)")
            
            # Convert tool calls to our task format
            tasks = []
            for i, tool_call in enumerate(message["tool_calls"]):
                func_name = tool_call["function"]["name"]
                args_str = tool_call["function"]["arguments"]
                
                # Try to parse arguments with error handling for malformed JSON
                try:
                    func_args = json.loads(args_str)
                except json.JSONDecodeError as e:
                    logger.error(f"[LLM_TOOL_{i+1}] Malformed JSON arguments for {func_name}: {e}")
                    logger.error(f"[LLM_TOOL_{i+1}] Raw arguments (first 500 chars): {args_str[:500]}")
                    
                    # Try aggressive cleaning and URL extraction
                    try:
                        # Strategy 1: Extract URL from garbage and reconstruct minimal valid JSON
                        url_match = re.search(r'https?://[^\s\'"}<]+', args_str)
                        if url_match and func_name == "parse_csv":
                            extracted_url = url_match.group(0)
                            # Remove any trailing garbage from URL
                            extracted_url = re.sub(r'[\'"}}]+.*$', '', extracted_url)
                            func_args = {"url": extracted_url, "path": ""}
                            logger.info(f"[LLM_TOOL_{i+1}] Extracted URL from garbage: {extracted_url}")
                        else:
                            # Strategy 2: Try standard JSON cleaning
                            # Remove trailing commas before closing braces/brackets
                            cleaned = re.sub(r',\s*}', '}', args_str)
                            cleaned = re.sub(r',\s*]', ']', cleaned)
                            # Remove control characters and non-ASCII garbage
                            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f\u0080-\uffff]', '', cleaned)
                            # Extract just the JSON object if there's garbage after
                            json_match = re.search(r'\{[^}]*\}', cleaned)
                            if json_match:
                                cleaned = json_match.group(0)
                            func_args = json.loads(cleaned)
                            logger.info(f"[LLM_TOOL_{i+1}] Successfully cleaned malformed JSON")
                    except Exception as repair_error:
                        # If still fails, skip this tool call
                        logger.error(f"[LLM_TOOL_{i+1}] Could not repair JSON: {repair_error}")
                        logger.error(f"[LLM_TOOL_{i+1}] Skipping tool call {func_name}")
                        continue
                
                # Clean URL arguments - remove trailing punctuation
                if "url" in func_args and isinstance(func_args["url"], str):
                    func_args["url"] = func_args["url"].rstrip('.,;:\"\' ')
                
                logger.info(f"[LLM_TOOL_{i+1}] {func_name} with args: {func_args}")
                
                tasks.append({
                    "id": f"task_{i+1}",
                    "tool_name": func_name,
                    "inputs": func_args,
                    "produces": [{"key": f"{func_name}_result_{i+1}", "type": "json"}],
                    "notes": f"Generated via function calling: {func_name}"
                })
            
            return {
                "tasks": tasks,
                "tool_calls": message["tool_calls"],
                "content": message.get("content"),
                "finish_reason": data["choices"][0]["finish_reason"]
            }
        else:
            # Model responded directly without tool calls
            logger.info(f"[LLM_DIRECT] Model responded without tool calls")
            return {
                "tasks": [],
                "tool_calls": None,
                "content": message.get("content"),
                "finish_reason": data["choices"][0]["finish_reason"]
            }


async def call_llm_for_plan(page_data: Dict[str, Any], previous_attempts: List[Any] = None) -> str:
    """Generate execution plan using LLM with correct schema and previous attempt context"""
    
    # Build context from previous attempts
    previous_context = ""
    if previous_attempts and len(previous_attempts) > 0:
        previous_context = "\n\nPREVIOUS ATTEMPTS (what didn't work):\n"
        for prev in previous_attempts:
            previous_context += f"\nAttempt {prev.attempt_number}:\n"
            previous_context += f"  - Answer submitted: {str(prev.answer)[:100]}\n"
            previous_context += f"  - Was correct: {prev.correct}\n"
            if prev.error:
                previous_context += f"  - Error: {prev.error}\n"
            if prev.submission_response:
                error_msg = prev.submission_response.get('error', prev.submission_response.get('reason', 'N/A'))
                previous_context += f"  - Server response: {error_msg}\n"
        previous_context += "\nUSE THIS INFORMATION TO GENERATE A BETTER PLAN FOR THIS ATTEMPT!\n"
    
    system_prompt = """
You are an execution planner for an automated quiz-solving agent. Your job is to analyze a rendered quiz page and generate a machine-executable JSON plan.

OUTPUT FORMAT - YOU MUST OUTPUT VALID JSON ONLY:
{
  "submit_url": "string - the URL to POST the answer to",
  "origin_url": "string - the original quiz page URL",
  "tasks": [
    {
      "id": "task_1",
      "tool_name": "tool_name_here",
      "inputs": {},
      "produces": [{"key": "output_key", "type": "string"}],
      "notes": "description"
    }
  ],
  "final_answer_spec": {
    "type": "boolean|number|string|json",
    "from": "artifact_key or literal_value"
  },
  "request_body": {
    "email_key": "email",
    "secret_key": "secret", 
    "url_value": "url",
    "answer_key": "answer"
  }
}

CRITICAL: The request_body format above is FIXED - copy it EXACTLY as shown!
- "email_key": "email" means the field name is "email" 
- "secret_key": "secret" means the field name is "secret"
- "url_value": "url" means the field name is "url" (NOT the actual URL!)
- "answer_key": "answer" means the field name is "answer" (NOT the actual answer value!)

DO NOT put actual values like URLs or answer numbers in request_body!

IMPORTANT RULES:
1. Submit URL must be extracted from the page text
2. Tasks array contains the work to do (can be empty if answer is direct)
3. final_answer_spec.from references an artifact key from tasks OR a literal value
4. request_body MUST be copied exactly as shown in the example above
5. Return ONLY valid JSON, no markdown wrapper, no extra text

AVAILABLE TOOLS:
- render_js_page(url): Render page and get text/links/code
- fetch_text(url): Fetch text content
- download_file(url): Download binary file
- parse_csv(path OR url): Parse CSV from local path or URL → produces {"dataframe_key": "df_0", ...}
- parse_excel/parse_json_file/parse_html_tables/parse_pdf_tables: Parse files
- dataframe_ops(op, params): DataFrame operations
  * params MUST include "dataframe_key" (e.g., "df_0" from parse_csv)
  * Example: {"op": "sum", "params": {"dataframe_key": "df_0", "column": "columnName"}}
- make_plot(spec): Create chart
- zip_base64(paths): Create zip archive
- call_llm(prompt, system_prompt, max_tokens, temperature): Call LLM

CRITICAL TASK CHAINING RULES:
- parse_csv creates a dataframe with key "df_0"
- To use that dataframe, you MUST reference it: {"dataframe_key": "df_0"}
- CSV files without headers have NUMERIC column names: "0", "1", "2", etc. (as strings!)
- When summing/filtering CSV data, use column "0" for the first column
- Example tasks for "sum numbers in CSV":
  Task 1: {"id": "task_1", "tool_name": "parse_csv", "inputs": {"url": "data.csv"}, "produces": [{"key": "csv_data", "type": "json"}]}
  Task 2: {"id": "task_2", "tool_name": "dataframe_ops", "inputs": {"op": "sum", "params": {"dataframe_key": "df_0", "column": "0"}}, "produces": [{"key": "sum_result", "type": "number"}]}
"""
    
    prompt = f"""
QUIZ PAGE DATA:
Text: {page_data['text']}
Code blocks: {page_data['code_blocks']}
Links: {page_data['links']}
Audio sources: {page_data.get('audio_sources', [])}
HTML preview: {page_data['html'][:500]}...{previous_context}

ANALYZE THIS QUIZ AND GENERATE THE EXECUTION PLAN JSON.

Requirements:
1. Find the submit URL (look for POST endpoint, /submit, /answer paths)
2. Identify what answer is required (read the instruction text)
3. IF AUDIO IS PRESENT: Add tasks to download and transcribe it FIRST to get instructions
4. Determine the answer type (boolean, number, string, json)
5. Plan minimal tasks needed (often zero if answer is direct)
6. Build the request_body structure with proper field names

OUTPUT ONLY THE JSON PLAN, NO OTHER TEXT.
"""
    
    OPEN_AI_BASE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
    API_KEY = os.getenv("API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    json_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 3000,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(OPEN_AI_BASE_URL, headers=headers, json=json_data, timeout=120)
        response.raise_for_status()
        data = response.json()
        plan_text = data["choices"][0]["message"]["content"]
        logger.info(f"[LLM_PLAN] Raw plan response: {plan_text[:500]}")
        return plan_text
