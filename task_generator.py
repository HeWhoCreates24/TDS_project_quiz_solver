"""
Next task generation using LLM function calling
Handles multi-step workflows and intelligent task chaining
"""
import os
import json
import re
import logging
from typing import Any, Dict, List
import httpx
from tool_definitions import get_tool_definitions

logger = logging.getLogger(__name__)

# Use same model as llm_client.py for consistency
model = "openai/gpt-4o-mini"


async def generate_next_tasks(plan_obj: Dict[str, Any], artifacts: Dict[str, Any], page_data: Dict[str, Any], completion_status: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate next batch of tasks using LLM with function calling"""
    
    # Build context about what we have and what we need
    artifacts_summary = {}
    transcription_text = None  # Cache transcription lookup
    operations_done = []  # Track what operations were completed (ALL types)
    latest_dataframe = None  # Track latest dataframe key
    latest_image = None  # Track latest processed image
    latest_api_response = None  # Track latest API response
    
    # FIRST PASS: Find best transcription (prioritize successful ones)
    for key, value in artifacts.items():
        if 'transcribe' in key.lower() and isinstance(value, dict):
            text = value.get('text', '')
            # Only use transcriptions with actual content (>20 chars, not error messages)
            if text and len(text) > 20 and "sorry" not in text.lower() and "cannot" not in text.lower():
                transcription_text = text
                logger.info(f"[TRANSCRIPTION_CACHE] Using successful transcription from {key}: {text[:100]}")
                break  # Use first successful one
    
    for key, value in artifacts.items():
        # Transcription already extracted in first pass
        
        # Track ALL operation types (generalized for multimodal)
        if isinstance(value, dict):
            # Data operations
            if 'dataframe_ops' in key:
                ops_done = f"✓ Filtered dataframe -> {value.get('dataframe_key', 'unknown')}"
                operations_done.append(ops_done)
                latest_dataframe = value.get('dataframe_key')
            if 'statistics' in value:
                operations_done.append("✓ Calculated statistics")
            
            # Vision/Image operations
            if 'vision_result' in value or 'ocr_text' in value:
                operations_done.append("✓ Processed image/PDF with vision/OCR")
                if 'image_path' in value:
                    latest_image = value.get('image_path')
            
            # Chart/Visualization operations
            if 'chart_path' in value or 'plot_path' in value:
                operations_done.append("✓ Generated chart/visualization")
            
            # API operations
            if 'api_response' in value or 'response_data' in value:
                operations_done.append("✓ Fetched data from API")
                latest_api_response = key
            
            # Scraping operations
            if 'scraped_data' in value or 'html_content' in value:
                operations_done.append("✓ Scraped website data")
            
            # Data transformations
            if 'transformed_data' in value:
                operations_done.append("✓ Transformed data")
        
        # Build summary with appropriate truncation
        if isinstance(value, str):
            artifacts_summary[key] = value[:200] if len(value) > 200 else value
        elif isinstance(value, dict):
            # Keep full text for instruction-bearing artifacts
            if 'transcribe' in key.lower() and 'text' in value:
                artifacts_summary[key] = {k: v if k == 'text' else str(v)[:100] for k, v in list(value.items())[:10]}
            elif 'vision_result' in value or 'ocr_text' in value:
                # Keep vision/OCR results full (often contain answer)
                artifacts_summary[key] = {k: v if k in ['vision_result', 'ocr_text'] else str(v)[:100] for k, v in list(value.items())[:10]}
            else:
                artifacts_summary[key] = {k: str(v)[:100] for k, v in list(value.items())[:5]}
        else:
            artifacts_summary[key] = str(value)[:200]
    
    # Add operation tracking metadata for LLM
    if operations_done:
        artifacts_summary['_COMPLETED_OPERATIONS'] = operations_done
    if latest_dataframe:
        artifacts_summary['_LATEST_DATAFRAME'] = latest_dataframe
    if latest_image:
        artifacts_summary['_LATEST_IMAGE'] = latest_image
    if latest_api_response:
        artifacts_summary['_LATEST_API_RESPONSE'] = latest_api_response
    
    # ========================================
    # FORCED CHECKS - Ensure critical preprocessing (GENERALIZED for all data types)
    # ========================================
    
    # 1. AUDIO: Force transcription if audio present but not transcribed
    has_transcription = transcription_text is not None
    if not has_transcription:
        has_audio_file = any('content_type' in str(v) and 'audio' in str(v).lower() for v in artifacts.values())
        if has_audio_file:
            audio_path = None
            for key, value in artifacts.items():
                if isinstance(value, dict) and value.get('content_type', '').startswith('audio'):
                    audio_path = value.get('path')
                    break
            
            if audio_path:
                logger.info(f"[FORCE_TRANSCRIBE] Audio file found but not transcribed. Forcing transcribe_audio")
                return [{
                    "id": f"forced_transcribe_{len(artifacts)}",
                    "tool_name": "transcribe_audio",
                    "inputs": {"audio_path": audio_path},
                    "produces": [{"key": f"transcribe_audio_result_{len(artifacts)}", "type": "json"}],
                    "notes": "Forced transcription of audio file"
                }]
    
    # 2. CSV: Force parsing if mentioned but not loaded
    has_dataframe = any('dataframe_key' in str(v) for v in artifacts.values())
    if not has_dataframe:
        combined_text = f"{transcription_text or ''} {page_data.get('text', '')}".lower()
        mentions_csv = 'csv' in combined_text or ('data' in combined_text and 'file' in combined_text)
        
        if mentions_csv:
            csv_link = None
            for link in page_data.get('links', []):
                if isinstance(link, str) and '.csv' in link.lower():
                    csv_link = link
                    break
            
            if csv_link:
                logger.info(f"[FORCE_PARSE_CSV] CSV mentioned but no dataframe found. Forcing parse_csv")
                return [{
                    "id": f"forced_parse_csv_{len(artifacts)}",
                    "tool_name": "parse_csv",
                    "inputs": {"url": csv_link},
                    "produces": [{"key": f"parse_csv_result_{len(artifacts)}", "type": "json"}],
                    "notes": "Forced CSV parsing"
                }]
    
    # 3. IMAGE/PDF: Force vision/OCR if present but not processed
    has_vision_result = any('vision' in str(k).lower() or 'ocr' in str(k).lower() for k in artifacts.keys())
    if not has_vision_result:
        image_or_pdf_path = None
        for key, value in artifacts.items():
            if isinstance(value, dict):
                content_type = value.get('content_type', '')
                if 'image' in content_type or 'pdf' in content_type:
                    image_or_pdf_path = value.get('path')
                    break
        
        if image_or_pdf_path:
            combined_text = f"{transcription_text or ''} {page_data.get('text', '')}".lower()
            vision_keywords = ['describe', 'identify', 'detect', 'recognize', 'extract text', 'ocr', 'read', 'image']
            if any(kw in combined_text for kw in vision_keywords):
                logger.info(f"[FORCE_VISION] Image/PDF found but not processed. Forcing vision/OCR")
                return [{
                    "id": f"forced_vision_{len(artifacts)}",
                    "tool_name": "analyze_image",
                    "inputs": {"image_path": image_or_pdf_path},
                    "produces": [{"key": f"vision_result_{len(artifacts)}", "type": "json"}],
                    "notes": "Forced vision/OCR analysis"
                }]
    
    # 4. JAVASCRIPT: Force JS rendering if HTML contains script tags but no rendered content
    for key, value in artifacts.items():
        if isinstance(value, dict) and 'content' in value:
            html_content = value.get('content', '')
            if isinstance(html_content, str) and '<script' in html_content:
                # Check if we already have a rendered version of this URL
                has_rendered = any('rendered_' in str(k) or 'scraped_' in str(k) for k in artifacts.keys())
                if not has_rendered:
                    # Extract URL from page_data or look for it in artifacts
                    target_url = None
                    for link in page_data.get('links', []):
                        if isinstance(link, str) and 'scrape' in link.lower():
                            target_url = link
                            break
                    
                    if target_url:
                        logger.info(f"[FORCE_RENDER_JS] HTML with <script> detected but no rendered content. Forcing render_js_page")
                        return [{
                            "id": f"forced_render_js_{len(artifacts)}",
                            "tool_name": "render_js_page",
                            "inputs": {"url": target_url},
                            "produces": [{"key": f"rendered_page_{len(artifacts)}", "type": "json"}],
                            "notes": "Forced JavaScript rendering to extract dynamic content"
                        }]
    

    system_prompt = """You are a task planner that MUST use function calling to specify next steps.

IMPORTANT CONSTRAINTS:
- You MUST call tools using function calling - do NOT respond with text explanations
- Keep reasoning minimal - just enough to track what you're doing
- Focus on ACTION, not explanation

CRITICAL: Check if the answer is ALREADY in the artifacts!
- If rendered_page_X has text like "Secret code is 1371" → NO MORE TOOLS NEEDED, answer is ready!
- If vision_result contains the extracted value → NO MORE TOOLS NEEDED, answer is ready!
- If statistics are already calculated → NO MORE TOOLS NEEDED, answer is ready!
- DO NOT make unnecessary API calls or fetch operations when the answer is already available
- DO NOT call submit endpoints - the system handles submission automatically

Understanding multi-step workflows (GENERALIZED for all data types):

DATA ANALYSIS workflows:
- When instructions mention filtering/selecting a subset, FIRST filter, THEN calculate
- Example: "sum values >= 100" requires TWO steps:
  1. dataframe_ops(op="filter", params={"dataframe_key": "df_0", "condition": "column >= 100"}) → creates df_1
  2. calculate_statistics(dataframe="df_1", stats=["sum"]) → calculates sum on filtered data

VISION/IMAGE workflows:
- Raw image/PDF → analyze_image or ocr_image → extract text/data → use in calculations

API workflows:
- fetch_api → extract relevant fields → transform/calculate as needed

SCRAPING workflows:
- fetch_text → might get HTML with <script> tags → WAIT for render_js_page (automatic)
- After render_js_page completes → check rendered text for answer
- If rendered text contains the answer (e.g., "Secret code is 1371") → STOP, no more tools needed
- DO NOT call POST requests to submit - that's handled automatically

MULTIMODAL workflows:
- Combine: audio transcription + image vision + data analysis as needed
- Chain results: output of one tool becomes input to next

Tool chaining patterns:
- parse_csv → dataframe (e.g., "df_0")
- dataframe_ops with filter → NEW dataframe (e.g., "df_1")  
- calculate_statistics → uses latest/appropriate dataframe
- analyze_image → vision_result → can be used in further processing
- fetch_text → rendered_page (automatic if JS detected) → answer extracted automatically"""

    # transcription_text already cached from above
    # Check what we have and what we're missing
    has_statistics = any('statistics' in str(v) for v in artifacts.values())
    has_dataframe = any('dataframe_key' in str(v) for v in artifacts.values())
    
    prompt = f"""
STATE:
Artifacts: {json.dumps(artifacts_summary, indent=2)}

Page text: {page_data.get('text', 'N/A')[:500]}
Links: {page_data.get('links', [])}

{f'''INSTRUCTIONS FROM AUDIO/TEXT:
"{transcription_text}"

Available tools for data operations:
- dataframe_ops(op="filter", params={{"dataframe_key": "df_X", "condition": "column >= value"}})
- calculate_statistics(dataframe="df_X", stats=["sum", "mean", "median", "count", etc.])

Current state: Transcription ✓, Data {"✓" if has_dataframe else "✗"}, Calculations {"✓" if has_statistics else "✗"}
''' if transcription_text else ''}

IMPORTANT RULES:
1. Check _COMPLETED_OPERATIONS to see what's already done - DO NOT repeat these operations
2. If a filter was already applied, use _LATEST_DATAFRAME for your next calculation
3. Each operation should be done ONCE - check artifacts before acting
4. **CRITICAL**: Check if answer is ALREADY AVAILABLE in artifacts:
   - If rendered_page has meaningful text (e.g., "Secret code is 1371") → NO MORE TOOLS NEEDED
   - If statistics are calculated → NO MORE TOOLS NEEDED
   - If vision_result has extracted data → NO MORE TOOLS NEEDED
   - DO NOT make redundant API calls or POST requests
   - The system handles answer extraction and submission automatically

TASK: Based on the instructions above, decide if you need to call ANY tools, or if the answer is already ready.
- If answer is ready in artifacts → DO NOT call any tools (respond with reasoning only)
- If more work needed → call the necessary tools using function calling

Use function calling ONLY when tools are actually needed.
Your reasoning should be brief. Focus on checking what's ALREADY DONE before calling new tools.
"""

    logger.info(f"[GENERATE_NEXT_TASKS] Transcription text: {transcription_text}")
    logger.info(f"[GENERATE_NEXT_TASKS] Artifacts summary keys: {list(artifacts_summary.keys())}")
    logger.info(f"[GENERATE_NEXT_TASKS] Has statistics: {has_statistics}, Has dataframe: {has_dataframe}")
    logger.info(f"[GENERATE_NEXT_TASKS] Full prompt sent to LLM:\n{prompt}")

    # Force tool usage if we have data but no calculations yet
    # This prevents the LLM from saying "no tasks needed" when calculations are still required
    # BUT: Don't force if we already have rendered text with likely answer
    has_rendered_answer = any(
        isinstance(v, dict) and 'text' in v and len(str(v.get('text', ''))) > 10
        for k, v in artifacts.items() 
        if 'rendered_' in k or 'vision_' in k or 'extracted_' in k
    )
    
    force_tool_usage = has_dataframe and not has_statistics and transcription_text and not has_rendered_answer
    tool_choice_param = "required" if force_tool_usage else "auto"
    logger.info(f"[GENERATE_NEXT_TASKS] Tool choice: {tool_choice_param} (force_reason: df={has_dataframe}, no_stats={not has_statistics}, has_instructions={bool(transcription_text)}, has_rendered_answer={has_rendered_answer})")

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
        "tool_choice": tool_choice_param,
        "temperature": 0,
        "max_tokens": 2000,
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OPEN_AI_BASE_URL, headers=headers, json=json_data, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            message = data["choices"][0]["message"]
            logger.info(f"[GENERATE_NEXT_TASKS] LLM response message: {message}")
            
            # Check if model wants to call tools
            if message.get("tool_calls"):
                logger.info(f"[NEXT_TASKS] Model requested {len(message['tool_calls'])} additional tool call(s)")
                
                # Convert tool calls to task format
                next_tasks = []
                for i, tool_call in enumerate(message["tool_calls"]):
                    func_name = tool_call["function"]["name"]
                    raw_args = tool_call["function"]["arguments"]
                    
                    # Clean up malformed JSON from o1 model (reasoning tokens leak into JSON)
                    # Only do minimal cleanup at JSON level - detailed cleanup happens on values later
                    cleaned_args = raw_args
                    
                    # Remove markdown code block markers if present
                    if '```' in cleaned_args:
                        cleaned_args = re.sub(r'```[a-z]*\??', '', cleaned_args)
                    
                    # Only fix obvious O1 contamination patterns at JSON structure level
                    # Don't touch valid nested JSON structures
                    if '}})' in cleaned_args or '}}}' in cleaned_args:
                        # Fix duplicated closing braces that break JSON structure
                        cleaned_args = re.sub(r'\}\}\}+', '}}', cleaned_args)
                        cleaned_args = re.sub(r'\}\}\)', '}', cleaned_args)
                    
                    try:
                        func_args = json.loads(cleaned_args)
                    except json.JSONDecodeError as e:
                        logger.error(f"[NEXT_TASK_{i+1}] Failed to parse arguments after cleanup: {e}")
                        logger.error(f"[NEXT_TASK_{i+1}] Raw: {raw_args}")
                        logger.error(f"[NEXT_TASK_{i+1}] Cleaned: {cleaned_args}")
                        # Try original as last resort
                        func_args = json.loads(raw_args)
                    
                    # Clean string values - remove trailing junk AND o1 reasoning contamination
                    for key, value in func_args.items():
                        if isinstance(value, str):
                            # Only apply aggressive cleaning if O1 contamination patterns detected
                            has_contamination = any(p in value for p in ['}}', '//Oops', '\nJk', 'craft properly', 'Let\'s rewrite'])
                            
                            if has_contamination:
                                # O1 contamination detected - clean aggressively
                                value = re.sub(r'\}\}+.*$', '', value)  # Remove }})...
                                value = re.sub(r'//\s*Oops.*$', '', value, flags=re.IGNORECASE)  # Remove //Oops...
                                value = re.sub(r'\n.*Jk.*$', '', value, flags=re.IGNORECASE)  # Remove Jk contamination
                                value = re.sub(r'\n.*craft.*$', '', value, flags=re.IGNORECASE)  # Remove "craft properly"
                                value = re.sub(r'\n.*rewrite.*$', '', value, flags=re.IGNORECASE)  # Remove rewrite attempts
                            
                            # Always apply safe cleaning
                            value = re.sub(r'```.*$', '', value)     # Remove code blocks
                            value = value.rstrip('.,;:"\' ?!')       # Remove trailing punctuation
                            func_args[key] = value
                    
                    # Recursively clean nested dicts (like params)
                    if "params" in func_args and isinstance(func_args["params"], dict):
                        for key, value in func_args["params"].items():
                            if isinstance(value, str):
                                # Same conservative approach for nested params
                                has_contamination = any(p in value for p in ['}}', '//Oops', '\nJk', 'craft properly', 'Let\'s rewrite'])
                                
                                if has_contamination:
                                    value = re.sub(r'\}\}+.*$', '', value)
                                    value = re.sub(r'//\s*Oops.*$', '', value, flags=re.IGNORECASE)
                                    value = re.sub(r'\n.*Jk.*$', '', value, flags=re.IGNORECASE)
                                    value = re.sub(r'\n.*craft.*$', '', value, flags=re.IGNORECASE)
                                    value = re.sub(r'\n.*rewrite.*$', '', value, flags=re.IGNORECASE)
                                
                                value = re.sub(r'```.*$', '', value)
                                value = value.rstrip('.,;:"\' ?!')
                                func_args["params"][key] = value
                    
                    logger.info(f"[NEXT_TASK_{i+1}] {func_name} with args: {func_args}")
                    
                    next_tasks.append({
                        "id": f"next_task_{i+1}",
                        "tool_name": func_name,
                        "inputs": func_args,
                        "produces": [{"key": f"{func_name}_result_{i+1}", "type": "json"}],
                        "notes": f"Follow-up task: {func_name}"
                    })
                
                return next_tasks
            else:
                # LLM decided no more tools needed - likely answer is already in artifacts
                logger.info(f"[NEXT_TASKS] LLM determined no additional tools needed - answer likely ready in artifacts")
                logger.info(f"[NEXT_TASKS] LLM reasoning: {message.get('content', 'No reasoning provided')}")
                return []
                
    except Exception as e:
        logger.error(f"[NEXT_TASKS] Error generating tasks: {e}")
        return []
