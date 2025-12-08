"""Completion checking logic for quiz execution
Fast deterministic checks + LLM-based evaluation
"""
import re
import json
import logging
from typing import Any, Dict, Optional, Tuple
from llm_client import call_llm

logger = logging.getLogger(__name__)

# Global statistics for completion check optimization
completion_check_stats = {
    "checks_performed": 0,
    "checks_skipped": 0,
    "llm_calls_avoided": 0,
    "fast_checks_used": 0,
    "time_saved_seconds": 0.0
}


def classify_operation(tool_name: str) -> str:
    """Classify operation based on generic semantic patterns.
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
        'call_llm', 'generate', 'report',                # LLM/generation (final step)
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


# Explicit overrides for all 24 known tools
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
    """Get operation type with override support.
    Checks explicit overrides first, then uses semantic classification.
    """
    # Check explicit overrides
    if tool_name in OPERATION_OVERRIDES:
        return OPERATION_OVERRIDES[tool_name]
    
    # Use semantic classification
    return classify_operation(tool_name)


def should_check_completion(
    last_task: Optional[Dict[str, Any]], 
    artifacts: Dict[str, Any]
) -> Tuple[bool, str]:
    """Determine if completion check is needed based on last operation.
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


def check_completion_fast(artifacts: Dict[str, Any], page_text: str = "", transcription_text: str = "") -> Optional[Dict[str, Any]]:
    """Fast deterministic checks for obvious completion signals - NO LLM CALL
    Generalized for all multimodal data types: data, images, PDFs, APIs, scraping, vision, etc.
    
    Returns:
        dict with {"complete": bool, "reason": str, "needs_more": bool} if certain
        None if uncertain and LLM evaluation needed
    """
    
    # 1. FINAL RESULT ARTIFACTS - definitely done!
    for artifact_key, artifact_value in artifacts.items():
        # Statistics (from data analysis)
        if isinstance(artifact_value, dict) and 'statistics' in artifact_value:
            stats_dict = artifact_value.get('statistics', {})
            if isinstance(stats_dict, dict):
                for col_stats in stats_dict.values():
                    if isinstance(col_stats, dict) and len(col_stats) > 0:
                        logger.info(f"[FAST_CHECK] Statistics found in {artifact_key}")
                        return {"complete": True, "reason": "statistics_present"}
        
        # Chart/visualization artifacts (images, interactive charts)
        if isinstance(artifact_value, dict):
            if 'chart_path' in artifact_value or 'plot_path' in artifact_value:
                logger.info(f"[FAST_CHECK] Chart/visualization found in {artifact_key}")
                return {"complete": True, "reason": "visualization_present"}
            
            # API response with final data
            if 'api_response' in artifact_value or 'response_data' in artifact_value:
                response_data = artifact_value.get('api_response') or artifact_value.get('response_data')
                if response_data and not isinstance(response_data, dict):  # Simple value
                    logger.info(f"[FAST_CHECK] API response value found in {artifact_key}")
                    return {"complete": True, "reason": "api_value_present"}
            
            # Vision/OCR results with extracted text
            if 'vision_result' in artifact_value or 'ocr_text' in artifact_value:
                text_result = artifact_value.get('vision_result') or artifact_value.get('ocr_text')
                if text_result and isinstance(text_result, str) and len(text_result.strip()) > 0:
                    logger.info(f"[FAST_CHECK] Vision/OCR result found in {artifact_key}")
                    return {"complete": True, "reason": "vision_result_present"}
            
            # Pattern extraction results (extract_patterns output)
            if 'pattern_type' in artifact_value and 'count' in artifact_value:
                # Check if page asks for count/number
                count_keywords = ['how many', 'count', 'number of', 'total']
                # Check if additional operations are needed (filtering, comparison, blacklist, etc.)
                multi_step_keywords = ['not in', 'blacklist', 'filter', 'exclude', 'except', 'remove', 'greater', 'less', 'compare']
                
                has_count_question = any(keyword in page_text.lower() for keyword in count_keywords)
                needs_additional_ops = any(keyword in page_text.lower() for keyword in multi_step_keywords)
                
                # Only fast-complete if it's a simple count question without filtering
                if has_count_question and not needs_additional_ops:
                    logger.info(f"[FAST_CHECK] Pattern extraction count found in {artifact_key}")
                    return {"complete": True, "reason": "pattern_count_present"}
    
    # 2. EXTRACTED VALUES - likely done! (but check for multi-step operations)
    extracted_artifacts = [k for k in artifacts.keys() if k.startswith('extracted_')]
    if extracted_artifacts:
        # Check if additional operations are needed (filtering, comparison, etc.)
        multi_step_keywords = ['not in', 'blacklist', 'filter', 'exclude', 'except', 'remove', 'greater', 'less', 'compare']
        needs_additional_ops = any(keyword in page_text.lower() for keyword in multi_step_keywords)
        
        if not needs_additional_ops:
            logger.info(f"[FAST_CHECK] Found extracted artifacts: {extracted_artifacts}")
            return {
                "complete": True,
                "reason": f"Extracted values found: {extracted_artifacts}",
                "needs_more": False
            }
    
    # 2b. RENDERED PAGES WITH TEXT - likely done for scraping tasks!
    rendered_artifacts = [k for k in artifacts.keys() if k.startswith('rendered_')]
    if rendered_artifacts:
        for key in rendered_artifacts:
            value = artifacts[key]
            if isinstance(value, dict) and 'text' in value:
                text_content = value.get('text', '')
                # If rendered text has substantial content, consider it complete
                if isinstance(text_content, str) and len(text_content.strip()) > 10:
                    logger.info(f"[FAST_CHECK] Rendered page with text found: {key}")
                    return {
                        "complete": True,
                        "reason": f"Rendered page with text content: {key}",
                        "needs_more": False
                    }
    
    # 3. INTERMEDIATE STATES - need more work
    combined_text = f"{page_text} {transcription_text}".lower()
    
    # Data analysis: detect filter-then-calculate pattern
    has_dataframe = any('dataframe_key' in str(v) for v in artifacts.values())
    has_filtered_df = any('dataframe_ops' in str(k) for k in artifacts.keys())
    has_statistics = any('statistics' in str(v) for v in artifacts.values())
    
    calc_keywords = ['sum', 'add', 'total', 'mean', 'average', 'count', 'multiply', 'calculate', 'aggregate']
    filter_keywords = ['greater', 'less', 'equal', 'filter', 'where', 'select', 'cutoff', 'threshold']
    
    needs_calculation = any(kw in combined_text for kw in calc_keywords)
    needs_filter = any(kw in combined_text for kw in filter_keywords)
    
    # Pattern 1: Have unfiltered data, need filter first
    if has_dataframe and not has_filtered_df and needs_filter and needs_calculation:
        logger.info("[FAST_CHECK] Have dataframe but need filtering before calculation")
        return {
            "complete": False,
            "reason": "Data loaded but filtering not applied yet",
            "needs_more": True
        }
    
    # Pattern 2: Have filtered data, need calculations
    if has_filtered_df and needs_calculation and not has_statistics:
        logger.info("[FAST_CHECK] Have filtered dataframe but need calculations")
        return {
            "complete": False,
            "reason": "Data filtered but calculations not performed yet",
            "needs_more": True
        }
    
    # Pattern 3: Have any dataframe, need calculations (general case)
    if has_dataframe and needs_calculation and not has_statistics:
        logger.info("[FAST_CHECK] Have dataframe but need calculations")
        return {
            "complete": False,
            "reason": "Data loaded but calculations not performed yet",
            "needs_more": True
        }
    
    # Image/PDF processing: have raw file but need processing
    has_image = any('image' in str(v).lower() or '.png' in str(v).lower() or '.jpg' in str(v).lower() for v in artifacts.values())
    has_pdf = any('.pdf' in str(v).lower() for v in artifacts.values())
    vision_keywords = ['describe', 'identify', 'detect', 'recognize', 'extract text', 'ocr', 'read']
    needs_vision = any(kw in combined_text for kw in vision_keywords)
    
    if (has_image or has_pdf) and needs_vision:
        vision_done = any('vision' in str(k).lower() or 'ocr' in str(k).lower() for k in artifacts.keys())
        if not vision_done:
            logger.info("[FAST_CHECK] Have image/PDF but need vision/OCR processing")
            return {
                "complete": False,
                "reason": "Image/PDF downloaded but not processed with vision/OCR",
                "needs_more": True
            }
    
    # Scraping: have HTML but need extraction
    has_html = any('html' in str(v).lower() or 'scraped' in str(k).lower() for k, v in artifacts.items())
    scrape_keywords = ['find', 'extract', 'scrape', 'get data from', 'parse page']
    needs_extraction = any(kw in combined_text for kw in scrape_keywords)
    
    if has_html and needs_extraction:
        # Check for extracted data OR rendered page with meaningful text
        extracted = any('extracted' in str(k).lower() for k in artifacts.keys())
        has_rendered_text = False
        for key, value in artifacts.items():
            if isinstance(value, dict) and 'text' in value:
                text_content = value.get('text', '')
                # Check if rendered text has meaningful content (not just whitespace/empty)
                if isinstance(text_content, str) and len(text_content.strip()) > 5:
                    has_rendered_text = True
                    break
        
        if not extracted and not has_rendered_text:
            logger.info("[FAST_CHECK] Have HTML but need extraction")
            return {
                "complete": False,
                "reason": "HTML scraped but data not extracted yet",
                "needs_more": True
            }
    
    # 4. Uncertain - need LLM to decide
    return None


async def check_plan_completion(
    plan_obj: Dict[str, Any], 
    artifacts: Dict[str, Any], 
    page_data: Dict[str, Any],
    last_executed_task: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Check if plan execution is complete - with smart skip logic, fast checks, and LLM fallback"""
    
    # PHASE 1: Smart skip logic - avoid unnecessary checks
    should_check, skip_reason = should_check_completion(last_executed_task, artifacts)
    
    if not should_check:
        logger.info(f"[COMPLETION_SKIP] Skipping completion check - reason: {skip_reason}")
        completion_check_stats["checks_skipped"] += 1
        completion_check_stats["llm_calls_avoided"] += 1
        completion_check_stats["time_saved_seconds"] += 1.0  # Estimate 1s per skipped check
        return {
            "answer_ready": False,
            "needs_more_tasks": True,
            "reason": f"Skipped check after {skip_reason}",
            "recommended_next_action": "Continue execution",
            "fast_check": True,
            "check_skipped": True
        }
    
    # Track that we're performing a check
    completion_check_stats["checks_performed"] += 1
    logger.info(f"[COMPLETION_CHECK] Performing check - reason: {skip_reason}")
    
    # PHASE 2: Try fast deterministic checks (no LLM call)
    page_text = page_data.get('text', '')
    
    # Extract transcription from artifacts if available (for audio quizzes)
    transcription_text = ""
    for key, value in artifacts.items():
        if 'transcription' in key.lower() and isinstance(value, str):
            transcription_text = value
            break
    
    fast_result = check_completion_fast(artifacts, page_text, transcription_text)
    
    if fast_result is not None:
        logger.info(f"[FAST_CHECK] Deterministic result: complete={fast_result['complete']}, reason={fast_result['reason']}")
        return {
            "answer_ready": fast_result["complete"],
            "needs_more_tasks": fast_result.get("needs_more", False),
            "reason": fast_result["reason"],
            "recommended_next_action": "Submit answer" if fast_result["complete"] else "Execute more tasks",
            "fast_check": True
        }
    
    # SECOND: If uncertain, use LLM for intelligent evaluation
    logger.info("[LLM_CHECK] Fast checks inconclusive, calling LLM for evaluation")
    
    # Quick check: if we have statistics results with actual values, we're done
    for artifact_key, artifact_value in artifacts.items():
        if isinstance(artifact_value, dict) and 'statistics' in str(artifact_key).lower():
            has_numeric = any(isinstance(v, (int, float)) for v in artifact_value.values())
            if has_numeric:
                logger.info(f"[LLM_CHECK] Skipping LLM - found numeric statistics: {artifact_key}")
                return {
                    "answer_ready": True,
                    "needs_more_tasks": False,
                    "reason": f"Statistics computed: {artifact_value}",
                    "recommended_next_action": "Submit answer",
                    "fast_check": False
                }
    
    system_prompt = """
        You are a strict task planner evaluating execution progress. Determine if the answer is ready to submit or if more tasks are needed.
        
        IMPORTANT: Only set answer_ready=true if you can ACTUALLY SEE the final answer value in the artifacts.
        Do NOT make assumptions or hallucinate values you cannot see.
        
        Respond with JSON:
        {
            "answer_ready": boolean,
            "needs_more_tasks": boolean,
            "reason": "explanation",
            "recommended_next_action": "what should be done next"
        }
        """
    
    artifacts_summary = {}
    for key, value in artifacts.items():
        if isinstance(value, (dict, list)):
            artifacts_summary[key] = str(value)[:200]
        else:
            artifacts_summary[key] = str(value)[:200]
    
    prompt = f"""
        Current Plan:
        {json.dumps(plan_obj.get('final_answer_spec', {}), indent=2)}
        
        Produced Artifacts:
        {json.dumps(artifacts_summary, indent=2)}
        
        Quiz Instructions:
        {page_data.get('text', 'N/A')[:1000]}
        
        SMART EVALUATION:
        1. The plan specifies we need artifact from "{plan_obj.get('final_answer_spec', {}).get('from', 'unknown')}"
        2. Check if that artifact exists AND is a meaningful value (not HTML, not script tags, not a dict)
        3. If NOT useful, check for alternatives:
           - Look for "extracted_*", "rendered_*", or "secret_*" artifacts
           - These might be better than the originally planned artifact
        4. If any meaningful value exists (string with actual data), answer_ready=true
        5. If only HTML/script/dicts exist, answer_ready=false
        
        Answer ready means: "We have a clean, meaningful value ready to submit, not HTML or code"
        """
    
    response_text = await call_llm(prompt, system_prompt, 1000, 0)
    
    try:
        cleaned_response = response_text.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response)
        cleaned_response = re.sub(r'\n?```$', '', cleaned_response)
        result = json.loads(cleaned_response)
        result["fast_check"] = False
        return result
    except json.JSONDecodeError:
        logger.error(f"[LLM_CHECK] Failed to parse LLM response: {response_text}")
        return {
            "answer_ready": False,
            "needs_more_tasks": True,
            "reason": "Could not parse LLM evaluation",
            "recommended_next_action": "Retry evaluation",
            "fast_check": False
        }


async def format_artifact(artifact_key: str, artifact_value: Any, page_instructions: str) -> Any:
    """Use LLM to extract meaningful information from complex artifacts"""
    
    # Handle extract_patterns results - return count when asking for "how many"
    if isinstance(artifact_value, dict) and 'pattern_type' in artifact_value and 'count' in artifact_value:
        # Check if question asks for count/number
        count_keywords = ['how many', 'count', 'number of', 'total']
        if any(keyword in page_instructions.lower() for keyword in count_keywords):
            logger.info(f"[ARTIFACT_FORMAT] Extracting count from pattern extraction: {artifact_value['count']}")
            return artifact_value['count']
    
    # Only process complex artifacts (dicts with HTML/text, long strings)
    should_process = False
    
    if isinstance(artifact_value, dict):
        # Check if dict contains HTML or text content
        if any(k in artifact_value for k in ['html', 'text', 'content', 'data']):
            should_process = True
    elif isinstance(artifact_value, str) and len(artifact_value) > 50:
        # Check if it's HTML or has tags
        if '<' in artifact_value or '>' in artifact_value:
            should_process = True
    
    if not should_process:
        return artifact_value
    
    logger.info(f"[ARTIFACT_FORMAT] Processing {artifact_key} to extract meaningful value")
    
    system_prompt = """You are an expert at extracting specific answers from scraped web content, rendered HTML, or structured data.

Your task: Extract the SPECIFIC ANSWER VALUE requested in the quiz instructions.

Rules:
- If the artifact contains the answer (a number, code, text value), extract ONLY that value
- Do NOT return HTML tags, formatting, or explanatory text
- Do NOT return the entire artifact
- If the answer is a number, return just the number
- If the answer is a code/secret, return just that code
- If you cannot find a specific answer, return the most relevant extracted text

Examples:
- Input: {"text": "Secret code is 1371 and not 1895"}
  Output: 1371

- Input: {"html": "<div>Answer: <strong>42</strong></div>"}
  Output: 42

- Input: {"text": "The result is abc123"}
  Output: abc123

Respond with ONLY the extracted value, nothing else."""

    artifact_str = json.dumps(artifact_value, indent=2) if isinstance(artifact_value, dict) else str(artifact_value)[:1000]
    
    prompt = f"""Quiz Instructions:
{page_instructions[:500]}

Artifact ({artifact_key}):
{artifact_str}

Extract the specific answer value needed based on the quiz instructions above.
Respond with ONLY the value, no explanation."""

    try:
        extracted = await call_llm(prompt, system_prompt, 500, 0)
        logger.info(f"[ARTIFACT_FORMAT] Extracted value: {extracted[:100]}")
        return extracted.strip()
    except Exception as e:
        logger.error(f"[ARTIFACT_FORMAT] Error formatting artifact: {e}")
        return artifact_value
