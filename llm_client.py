"""LLM client functions for making API calls to language models
Handles plan generation, tool calling, and general LLM interactions

CENTRAL LLM FUNCTION:
All LLM calls go through call_llm() which handles both text and tool-based interactions.
This ensures consistency and prevents tool schema drift.
"""
import os
import re
import json
import logging
from typing import Any, Dict, List, Optional
import httpx
from tool_definitions import get_tool_definitions, get_tool_usage_examples

logger = logging.getLogger(__name__)

# Use a non-reasoning model for reliable JSON output
# Reasoning models (gpt-5-nano) consume tokens on reasoning, leaving no content
model = "openai/gpt-4o-mini"  # Fast, reliable, and cost-effective


async def call_llm(
    prompt: str, 
    system_prompt: str = None, 
    max_tokens: int = 2000, 
    temperature: float = 0,
    use_tools: bool = False,
    tool_choice: str = "auto"
) -> Any:
    """
    CENTRAL LLM FUNCTION - All LLM interactions go through here.
    
    Args:
        prompt: User prompt
        system_prompt: System prompt (defaults to generic assistant)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        use_tools: Whether to include tool definitions (function calling)
        tool_choice: Tool choice mode ("auto", "required", "none")
    
    Returns:
        - If use_tools=False: String response from LLM
        - If use_tools=True: Full message object (may contain tool_calls)
    """
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
    
    # Add tool definitions if requested
    if use_tools:
        tools = get_tool_definitions()
        json_data["tools"] = tools
        json_data["tool_choice"] = tool_choice
        logger.info(f"[LLM_CALL] Using tools: {[t['function']['name'] for t in tools]}")
        logger.info(f"[LLM_CALL] Tool choice mode: {tool_choice}")
    
    # Reduced logging - only show truncated prompts
    logger.debug(f"[LLM_CALL] System prompt: {system_prompt[:100]}...")
    logger.debug(f"[LLM_CALL] User prompt: {prompt[:200]}...")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(OPEN_AI_BASE_URL, headers=headers, json=json_data, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        if use_tools:
            # Return full message object for tool calling
            message = data["choices"][0]["message"]
            if message.get("tool_calls"):
                logger.info(f"[LLM_RESPONSE] Tool calls: {len(message['tool_calls'])}")
            else:
                logger.debug(f"[LLM_RESPONSE] Text response: {message.get('content', '')[:100]}...")
            return message
        else:
            # Return just the text content
            answer_text = data["choices"][0]["message"]["content"]
            logger.debug(f"[LLM_RESPONSE] Content: {answer_text[:100]}...")
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
- Page has images requiring OCR → CALL download_file(url) AND analyze_image with task="ocr"
- Page references CSV/data and asks for sum/stats → CALL parse_csv(url) AND calculate_statistics or dataframe_ops
- Page has complex text → CALL call_llm(prompt) to extract information

NEVER call fetch_from_api or any tool to POST to /submit - we handle submissions automatically.

IMAGE/OCR EXAMPLES:
✅ "Extract number from image" → CALL download_file, THEN analyze_image with task="ocr"
✅ "Read text from image" → CALL download_file, THEN analyze_image with task="ocr"

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
    
    logger.info(f"[MULTIMODAL_DETECTION] has_images={has_images}, has_audio={has_audio}, has_csv={has_csv_link}")
    logger.info(f"[PAGE_TEXT_CHECK] Text: {page_data['text'][:300]}")
    
    data_hint = ""
    tool_choice_mode = "auto"
    
    # Infrastructure: Ensure image data is made available for LLM to work with
    if has_images:
        image_url = page_data['image_sources'][0]
        logger.info(f"[IMAGE_DETECTED] Page has image source: {image_url}")
        logger.info(f"[PAGE_TEXT] Full text: {page_data['text']}")
        
        # Inform LLM about available image - let it decide what to do
        data_hint = f"""

📷 IMAGE DATA AVAILABLE:
The page contains an image file: {image_url}

To access the image:
1. download_file to get the image: {{"url": "{image_url}"}}
2. analyze_image to process it: {{"image_path": "image_data", "task": "ocr"}}

Note: Use analyze_image for visual analysis, not call_llm (call_llm cannot process images).
"""
    # Infrastructure: Ensure audio/data is made available for LLM to work with
    elif has_audio:
        audio_url = page_data['audio_sources'][0]
        logger.info(f"[AUDIO_DETECTED] Page has audio source: {audio_url}")
        logger.info(f"[PAGE_TEXT] Full text: {page_data['text']}")
        logger.info(f"[MULTIMEDIA] Audio sources: {page_data.get('audio_sources', [])}")
        
        csv_link = None
        if has_csv_link:
            csv_link = next((link for link in page_data['links'] if '.csv' in str(link).lower()), None)
        
        data_hint = f"""

🎵 AUDIO DATA AVAILABLE:
The page contains an audio file: {audio_url}
{f'The page also contains a CSV file: {csv_link}' if csv_link else ''}

To access the audio:
1. download_file to get the audio: {{"url": "{audio_url}"}}
2. transcribe_audio to convert to text: {{"audio_path": "audio_data"}}

The audio may contain instructions for what to do with the data. Let the LLM interpret the transcription.
"""
    elif has_csv_link:
        csv_link = next((link for link in page_data['links'] if '.csv' in str(link).lower()), None)
        logger.info(f"[CSV_DETECTED] Page has CSV link: {csv_link}")
        logger.info(f"[PAGE_TEXT] Full text: {page_data['text']}")
        
        # Infrastructure: Inform LLM about CSV data availability
        data_hint = f"""

📊 CSV DATA AVAILABLE:
The page contains a CSV file: {csv_link}

To access the data, use parse_csv: {{"url": "{csv_link}"}}
Then interpret the page instructions to determine what analysis is needed.
"""
    
    prompt = f"""
QUIZ PAGE DATA:
Text: {page_data['text']}
Code blocks: {page_data['code_blocks']}
Links: {page_data['links']}
HTML preview: {page_data['html'][:500]}...{previous_context}{data_hint}

TASK: Determine what tools (if any) are needed to get the complete answer.

GUIDANCE:
- If the page says the answer can be "anything" or "any value", no tools needed
- If there's data to analyze (images, audio, CSV, APIs), use appropriate tools to access it
- Data loading tools: download_file, parse_csv, parse_excel, render_js_page, fetch_from_api
- Analysis tools: analyze_image, transcribe_audio, calculate_statistics, dataframe_ops, call_llm
- Interpret the page instructions to decide which tools and operations are needed

Remember: You can call multiple tools in ONE response. We execute them in sequence.
"""

    OPEN_AI_BASE_URL = os.getenv("LLM_BASE_URL", "https://aipipe.org/openrouter/v1/chat/completions")
    API_KEY = os.getenv("API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Get tool definitions
    tools = get_tool_definitions()
    
    # Log the full prompt and tools being sent to LLM
    logger.debug(f"[LLM_PLAN_PROMPT] System: {system_prompt[:200]}...")
    logger.debug(f"[LLM_PLAN_PROMPT] User prompt length: {len(prompt)} chars")
    logger.info(f"[LLM_PLAN_TOOLS] Available tools: {[t['function']['name'] for t in tools]}")
    logger.info(f"[LLM_PLAN_TOOLS] Tool choice mode: {tool_choice_mode}")
    
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
    """
    Generate execution plan using text-based JSON generation.
    Uses centralized call_llm() with tool documentation in prompts.
    """
    
    # Check for multimodal content
    page_text_lower = page_data.get('text', '').lower()
    image_sources = page_data.get('image_sources', [])
    audio_sources = page_data.get('audio_sources', [])
    links = page_data.get('links', [])
    
    has_images = len(image_sources) > 0
    has_audio = len(audio_sources) > 0
    has_csv = any('.csv' in str(link).lower() for link in links)
    
    logger.info(f"[PLAN_MULTIMODAL] has_images={has_images}, has_audio={has_audio}, has_csv={has_csv}")
    logger.info(f"[PLAN_MULTIMODAL] image_sources={image_sources}, audio_sources={audio_sources}")
    
    # Build multimodal hints
    multimodal_hint = ""
    if has_images and any(kw in page_text_lower for kw in ['ocr', 'read', 'extract', 'number', 'text', 'image']):
        image_url = image_sources[0]
        logger.info(f"[PLAN_IMAGE_DETECTED] Image: {image_url}")
        multimodal_hint = f"""

🖼️ IMAGE OCR DETECTED:
The page has an image: {image_url}
For OCR tasks, use:
1. download_file to get the image
2. analyze_image with task="ocr" to extract text/numbers
Do NOT use call_llm for image analysis - it cannot process images!
"""
    elif has_audio:
        audio_url = audio_sources[0]
        logger.info(f"[PLAN_AUDIO_DETECTED] Audio: {audio_url}")
        multimodal_hint = f"""

🎵 AUDIO DETECTED:
The page has audio: {audio_url}
Use download_file + transcribe_audio to get instructions from the audio.
"""
    
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
    
    system_prompt = f"""
You are an execution planner for an automated quiz-solving agent. Your job is to analyze a rendered quiz page and generate a machine-executable JSON plan.

OUTPUT FORMAT - YOU MUST OUTPUT VALID JSON ONLY:

JSON STRUCTURE:
{{
  "submit_url": "<URL to POST the answer - if page says 'POST to X', use X; if page only mentions 'url=Y', default to /submit endpoint>",
  "origin_url": "<the quiz URL>",
  "tasks": [
    {{
      "id": "task_1",
      "tool_name": "<tool_name>",
      "inputs": {{<tool parameters>}},
      "produces": [{{"key": "<artifact_name>"}}],  ← MUST be array of objects with "key" field
      "notes": "<what this task does>"
    }}
  ],
  "final_answer_spec": {{
    "type": "<string|number|boolean>",
    "from": "<artifact_key or literal value>"
  }},
  "request_body": {{
    "email_key": "email",
    "secret_key": "secret",
    "url_value": "url",
    "answer_key": "answer"
  }}
}}

CRITICAL RULES:

1. ARTIFACT REFERENCES:
   - Use {{{{artifact_name}}}} syntax to reference previous task outputs
   - Example: task_1 produces "csv_data" → task_2 uses "{{{{csv_data}}}}"
   - NEVER hardcode "df_0" - always use {{{{artifact}}}} pattern
   - System auto-extracts dataframe_key from artifacts

2. NESTED FIELD ACCESS:
   - Use dot notation for nested fields: "artifact.field.subfield"
   - Example: "api_result.data.secret_code" extracts nested value
   - Example: "chart_result.unique_categories" gets count from chart

3. TOOL NAMING:
   - Use EXACT tool names from documentation below
   - fetch_from_api (NOT "call_api")
   - create_chart (NOT "make_chart")
   - dataframe_ops (NOT "transform_data" for row operations)

4. REQUEST_BODY FORMAT (COPY EXACTLY):
   - "email_key": "email" (the field name, not actual email)
   - "secret_key": "secret" (the field name, not actual secret)
   - "url_value": "url" (the field name, not actual URL)
   - "answer_key": "answer" (the field name, not actual answer)
   - DO NOT put actual values in request_body!

5. WHEN NO TOOLS NEEDED:
   - tasks: []
   - final_answer_spec.from: "<actual answer value>"
   - Examples: literal strings, computed numbers, obvious answers

6. WHEN TOOLS NEEDED:
   - Create task chain with artifact references
   - final_answer_spec.from: "<artifact_key>" or "<artifact.field>"
   - Use dot notation for nested extraction

7. CALL_LLM PROMPTS - CRITICAL FOR CALCULATIONS:
   - When using call_llm, include ALL relevant instructions from the quiz text
   - If quiz provides formulas, include them VERBATIM in the call_llm prompt
   - If quiz specifies calculation methods, include them EXACTLY in the prompt
   - For mathematical calculations, REQUIRE STEP-BY-STEP WORK to prevent arithmetic errors
   - ALWAYS specify exact output format (JSON structure, field names, precision)
   
   Example for F1 score calculation:
   Quiz text: "using formula F1 = 2*tp / (2*tp + fp + fn), then average across labels"
   
   call_llm prompt MUST include:
   "Calculate F1 scores using THIS EXACT FORMULA: F1 = 2*tp / (2*tp + fp + fn)
   
   For EACH run in {{{{f1_data}}}}:
   1. Calculate F1 for label 'x': F1_x = 2*tp / (2*tp + fp + fn)
   2. Calculate F1 for label 'y': F1_y = 2*tp / (2*tp + fp + fn)
   3. Calculate macro-F1 = (F1_x + F1_y) / 2
   4. Round to 4 decimal places
   
   Show step-by-step arithmetic for each calculation.
   
   Then find which run has the HIGHEST macro-F1.
   
   Return ONLY this JSON (no explanations):
   {{'run_id': 'runX', 'macro_f1': 0.XXXX}}
   
   Where runX is the run with highest macro-F1 score."
   
   system_prompt: "You are a calculator. Follow formulas exactly. Show all arithmetic steps. Return ONLY the final JSON result with the highest macro-F1 run.
   
   For EACH run, show your work step-by-step:
   1. For EACH label (x, y), calculate: F1 = 2*tp / (2*tp + fp + fn)
      - Write out the substitution: F1_x = 2*7 / (2*7 + 1 + 3) = 14/18 = 0.7778
      - Write out the substitution: F1_y = 2*9 / (2*9 + 2 + 1) = 18/21 = 0.8571
   2. Calculate macro-F1: (F1_x + F1_y) / 2 = (0.7778 + 0.8571) / 2 = 0.8175
   3. Round to 4 decimal places
   
   CRITICAL: Show ALL arithmetic steps to avoid calculation errors!"
   
   - The call_llm prompt should be SELF-CONTAINED with all necessary instructions
   - For multi-step calculations, require showing work at EACH step

8. WHEN QUIZ CONTAINS FORMULAS - CRITICAL:
   Search the quiz text for mathematical formulas (patterns: "formula", "using", "=", "calculate").
   
   IF FOUND: The call_llm prompt is YOUR ONLY CHANCE to tell the LLM how to calculate correctly.
   
   MANDATORY PROMPT STRUCTURE:
   "Here is the data: {{{{artifact_name}}}}
   
   Calculate using THIS EXACT FORMULA from the quiz: <copy formula verbatim>
   
   For EACH item:
   1. Substitute actual values: formula = 2*<actual_tp> / (2*<actual_tp> + <actual_fp> + <actual_fn>)
   2. Show arithmetic: 2*7 / (2*7 + 1 + 3) = 14/18 = 0.7778
   3. Round to <N> decimal places
   
   <If question asks for maximum/minimum>: Compare all results, find highest/lowest
   
   Return ONLY JSON: {{'field1': value1, 'field2': value2}}
   No explanations, no markdown, just the JSON."
   
   WHY THIS MATTERS: LLMs make arithmetic errors. Forcing them to show work step-by-step catches mistakes.

{get_tool_usage_examples()}
"""
    
    prompt = f"""
QUIZ PAGE DATA:
Text: {page_data['text']}
Code blocks: {page_data['code_blocks']}
Links: {page_data['links']}
Audio sources: {page_data.get('audio_sources', [])}
Image sources: {page_data.get('image_sources', [])}
HTML preview: {page_data['html'][:500]}...{multimodal_hint}{previous_context}

QUIZ TEXT ANALYSIS:
Does the quiz text contain a mathematical formula? Search for: "formula", "using", "calculate using", "=" in equations.

IF YES - THIS IS CRITICAL:
The formula is: [extract the exact formula from quiz text]
Your call_llm prompt MUST be:
"Calculate using THIS EXACT FORMULA: [formula copied verbatim]

Data: {{{{{{{{artifact_name}}}}}}}}

For EACH item, show step-by-step:
1. Substitute values: [formula] = 2*[actual_value] / (2*[actual_value] + [fp_value] + [fn_value])
2. Calculate: numerator/denominator = result
3. Round to [N] decimal places

[If finding max/min]: Compare ALL results, return the highest/lowest.

Return ONLY this JSON: {{'field': value}}
No explanations."

GENERATE THE EXECUTION PLAN JSON:

Requirements:
1. Extract the submit URL from the page text
2. Identify what answer is required
3. IF AUDIO: Download and transcribe FIRST
4. IF FORMULA DETECTED ABOVE: Use the mandatory prompt template in call_llm
5. Determine answer type
6. Plan minimal tasks

OUTPUT ONLY THE JSON PLAN.
"""
    
    # Use centralized call_llm
    plan_text = await call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=3000,
        temperature=0,
        use_tools=False  # Text-based JSON generation
    )
    
    logger.debug(f"[LLM_PLAN] Raw plan response: {plan_text[:200]}...")
    return plan_text
