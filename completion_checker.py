"""Completion checking logic for quiz execution
Fast deterministic checks + LLM-based evaluation
"""
import re
import json
import logging
from typing import Any, Dict, Optional
from llm_client import call_llm

logger = logging.getLogger(__name__)


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
                if text_result and isinstance(text_result, str) and len(text_result) > 10:
                    logger.info(f"[FAST_CHECK] Vision/OCR text found in {artifact_key}")
                    return {"complete": True, "reason": "vision_text_present"}
    
    # 2. EXTRACTED VALUES - likely done!
    extracted_artifacts = [k for k in artifacts.keys() if k.startswith('extracted_')]
    if extracted_artifacts:
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


async def check_plan_completion(plan_obj: Dict[str, Any], artifacts: Dict[str, Any], page_data: Dict[str, Any]) -> Dict[str, Any]:
    """Check if plan execution is complete - uses fast checks first, LLM only when needed"""
    
    # FIRST: Try fast deterministic checks (no LLM call)
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
