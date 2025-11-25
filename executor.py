"""
Main execution orchestration for quiz solving
Handles plan execution, tool routing, and quiz pipeline
"""
import os
import time
import json
import logging
import re
import asyncio
from typing import Any, Dict, List
from dotenv import load_dotenv

# Import from refactored modules
from tool_executors import (
    dataframe_registry, get_secret, extract_json_from_markdown,
    render_page, fetch_text, download_file, parse_csv_async, parse_excel,
    parse_json_file, parse_html_tables, parse_pdf_tables,
    dataframe_ops, make_plot, zip_base64, answer_submit, extract_patterns
)
from llm_client import call_llm, call_llm_for_plan  # Centralized LLM calls
from completion_checker import check_plan_completion, format_artifact, log_completion_stats
from task_generator import generate_next_tasks
from models import QuizAttempt, QuizRun
from tools import ToolRegistry, ScrapingTools, CleansingTools, ProcessingTools, AnalysisTools, VisualizationTools
from parallel_executor import build_dependency_graph
from cache_manager import quiz_cache

load_dotenv()

logger = logging.getLogger(__name__)


async def execute_plan(plan_obj: Dict[str, Any], email: str, origin_url: str, page_data: Dict[str, Any] = None, quiz_attempt: QuizAttempt = None) -> Dict[str, Any]:
    """Execute the LLM-generated plan with iterative task batches"""
    try:
        plan = plan_obj
        artifacts = {}
        execution_log = []
        all_tasks = plan.get("tasks", [])
        max_iterations = 10
        iteration = 0
        
        if quiz_attempt:
            quiz_attempt.plan = plan
        
        logger.info(f"Starting execution with iterative plan refinement (max {max_iterations} iterations)")
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"=== Iteration {iteration} ===")
            logger.info(f"Executing {len(all_tasks)} tasks")
            
            # Track last executed task for completion check optimization
            last_executed_task = None
            
            # Build dependency graph and group tasks into parallel execution waves
            waves = build_dependency_graph(all_tasks, artifacts)
            
            # Execute tasks wave by wave
            for wave_idx, wave_tasks in enumerate(waves):
                if len(wave_tasks) > 1:
                    logger.info(f"[PARALLEL] Wave {wave_idx + 1}/{len(waves)}: Executing {len(wave_tasks)} tasks in parallel")
                    
                    # Check if all tasks in wave are I/O-bound (can be parallelized)
                    io_tools = ['download_file', 'fetch_text', 'render_js_page', 'scrape_with_javascript', 'fetch_from_api']
                    all_io_bound = all(task["tool_name"] in io_tools for task in wave_tasks)
                    
                    if all_io_bound:
                        logger.info(f"[PARALLEL] All tasks are I/O-bound - running concurrently")
                        # Execute I/O tasks in parallel
                        async def execute_io_task(task):
                            task_id = task["id"]
                            tool_name = task["tool_name"]
                            inputs = task.get("inputs", {})
                            
                            if tool_name == "download_file":
                                logger.info(f"[TOOL_EXEC] {task_id}: download_file - URL: {inputs['url']}")
                                result = await download_file(inputs["url"])
                                logger.info(f"[TOOL_RESULT] {task_id}: download_file - File size: {result.get('size')} bytes")
                                return (task, result)
                            elif tool_name == "fetch_text":
                                logger.info(f"[TOOL_EXEC] {task_id}: fetch_text - URL: {inputs['url']}")
                                result = await fetch_text(inputs["url"])
                                logger.info(f"[TOOL_RESULT] {task_id}: fetch_text - Text length: {len(result.get('text', ''))}")
                                return (task, result)
                            elif tool_name == "render_js_page":
                                logger.info(f"[TOOL_EXEC] {task_id}: render_js_page - URL: {inputs['url']}")
                                result = await render_page(inputs["url"])
                                logger.info(f"[TOOL_RESULT] {task_id}: render_js_page - HTML length: {len(result.get('html', ''))}")
                                return (task, result)
                            elif tool_name == "scrape_with_javascript":
                                result = await ScrapingTools.scrape_with_javascript(inputs["url"], inputs.get("wait_for_selector"), inputs.get("timeout", 30000))
                                return (task, {"content": result})
                            elif tool_name == "fetch_from_api":
                                result = await ScrapingTools.fetch_from_api(inputs["url"], inputs.get("method", "GET"), inputs.get("headers"), inputs.get("body"), inputs.get("timeout", 30))
                                return (task, result)
                        
                        # Run all I/O tasks concurrently
                        results = await asyncio.gather(*[execute_io_task(task) for task in wave_tasks])
                        
                        # Process results and store artifacts
                        for task, result in results:
                            task_id = task["id"]
                            produces = task.get("produces", [])
                            
                            # Track last executed task
                            last_executed_task = task
                            
                            # Store artifacts (same logic as sequential)
                            for produce in produces:
                                key = produce["key"]
                                if isinstance(result, dict) and key in result:
                                    artifacts[key] = result[key]
                                elif isinstance(result, dict):
                                    artifacts[key] = result
                                    logger.info(f"[ARTIFACT_STORAGE] Stored dict result for {key}: {str(result)[:200]}")
                                else:
                                    artifacts[key] = str(result)
                            
                            execution_log.append({
                                "task_id": task_id,
                                "status": "success",
                                "tool": task["tool_name"],
                                "produces": [p["key"] for p in produces],
                                "iteration": iteration
                            })
                    else:
                        # Mixed tools - fall back to sequential
                        logger.info(f"[PARALLEL] Mixed tool types - executing sequentially")
                        # Don't use pass - fall through to sequential logic below
                else:
                    logger.info(f"[PARALLEL] Wave {wave_idx + 1}/{len(waves)}: Executing {len(wave_tasks)} task")
                
                # Sequential execution logic (for single tasks or mixed tool types)
                if len(wave_tasks) == 1 or not all_io_bound:
                    for task in wave_tasks:
                        task_id = task["id"]
                        tool_name = task["tool_name"]
                        inputs = task.get("inputs", {})
                        produces = task.get("produces", [])
                        
                        # Track last executed task
                        last_executed_task = task
                        
                        logger.info(f"Executing task {task_id}: {tool_name}")
                        
                        try:
                            # GENERIC INFRASTRUCTURE: Resolve {{artifact}} references in ALL tool inputs
                            # This is a fallback - prompts should teach LLM to avoid this pattern
                            resolved_inputs = {}
                            for input_key, input_value in inputs.items():
                                if isinstance(input_value, str) and ('{{' in input_value or (input_value.startswith('{') and input_value.endswith('}') and ' ' not in input_value)):
                                    # Extract artifact name from {{artifact_name}} or {artifact_name}
                                    artifact_ref = input_value.strip('{}').strip()
                                    if artifact_ref in artifacts:
                                        artifact_data = artifacts[artifact_ref]
                                        # Extract content from dict artifacts
                                        if isinstance(artifact_data, dict):
                                            if 'path' in artifact_data:
                                                # For file artifacts (images, audio, PDFs, etc.)
                                                resolved_inputs[input_key] = artifact_data['path']
                                            elif 'content' in artifact_data:
                                                resolved_inputs[input_key] = artifact_data['content']
                                            elif 'text' in artifact_data:
                                                resolved_inputs[input_key] = artifact_data['text']
                                            elif 'dataframe_key' in artifact_data:
                                                # Extract dataframe_key for dataframe operations
                                                resolved_inputs[input_key] = artifact_data['dataframe_key']
                                            else:
                                                resolved_inputs[input_key] = artifact_data
                                        else:
                                            resolved_inputs[input_key] = artifact_data
                                        logger.info(f"[ARTIFACT_RESOLVE] Resolved {{{{{artifact_ref}}}}} in {tool_name}.{input_key}")
                                    else:
                                        resolved_inputs[input_key] = input_value
                                else:
                                    resolved_inputs[input_key] = input_value
                            
                            # Use resolved inputs
                            inputs = resolved_inputs
                            
                            # Tool execution routing
                            if tool_name == "render_js_page":
                                logger.info(f"[TOOL_EXEC] {task_id}: render_js_page - URL: {inputs['url']}")
                                result = await render_page(inputs["url"])
                                logger.info(f"[TOOL_RESULT] {task_id}: render_js_page - HTML length: {len(result.get('html', ''))}")
                            elif tool_name == "fetch_text":
                                logger.info(f"[TOOL_EXEC] {task_id}: fetch_text - URL: {inputs['url']}")
                                result = await fetch_text(inputs["url"])
                                logger.info(f"[TOOL_RESULT] {task_id}: fetch_text - Text length: {len(result.get('text', ''))}")
                            elif tool_name == "download_file":
                                logger.info(f"[TOOL_EXEC] {task_id}: download_file - URL: {inputs['url']}")
                                result = await download_file(inputs["url"])
                                logger.info(f"[TOOL_RESULT] {task_id}: download_file - File size: {result.get('size')} bytes")
                            elif tool_name == "analyze_image":
                                from tool_executors import analyze_image
                                image_path = inputs["image_path"]
                                logger.info(f"[TOOL_EXEC] {task_id}: analyze_image - Task: {inputs['task']}, Image: {image_path}")
                                result = await analyze_image(image_path, inputs["task"])
                                logger.info(f"[TOOL_RESULT] {task_id}: analyze_image - Result: {result.get('result', '')[:200]}...")
                            elif tool_name == "extract_audio_metadata":
                                from tools import MultimediaTools
                                audio_path = inputs["audio_path"]
                                logger.info(f"[TOOL_EXEC] {task_id}: extract_audio_metadata - Path: {audio_path}")
                                result = MultimediaTools.extract_audio_metadata(audio_path)
                                logger.info(f"[TOOL_RESULT] {task_id}: extract_audio_metadata - {result}")
                            elif tool_name == "transcribe_audio":
                                from tools import MultimediaTools
                                logger.info(f"[TOOL_EXEC] {task_id}: transcribe_audio - Path: {inputs.get('audio_path')}")
                                audio_path = inputs.get("audio_path")
                                result = await MultimediaTools.transcribe_audio(audio_path)
                                logger.info(f"[TOOL_RESULT] {task_id}: transcribe_audio - Text: {result.get('text', '')[:200]}...")
                            elif tool_name == "parse_csv":
                                logger.info(f"[TOOL_EXEC] {task_id}: parse_csv - Path: {inputs.get('path')}, URL: {inputs.get('url')}")
                                result = await parse_csv_async(
                                    path=inputs.get("path"),
                                    url=inputs.get("url")
                                )
                                logger.info(f"[TOOL_RESULT] {task_id}: parse_csv - DataFrame: {result.get('dataframe_key')}, Shape: {result.get('shape')}")
                            elif tool_name == "parse_excel":
                                result = parse_excel(inputs["path"])
                            elif tool_name == "parse_json_file":
                                result = parse_json_file(inputs["path"])
                            elif tool_name == "parse_html_tables":
                                result = parse_html_tables(inputs["path_or_html"])
                            elif tool_name == "parse_pdf_tables":
                                result = parse_pdf_tables(inputs["path"], inputs.get("pages", "all"))
                            elif tool_name == "extract_patterns":
                                logger.info(f"[TOOL_EXEC] {task_id}: extract_patterns - Pattern: {inputs.get('pattern_type')}")
                                text = inputs["text"]
                                # Resolve artifact reference
                                if isinstance(text, str) and text in artifacts:
                                    text = artifacts[text]
                                result = extract_patterns(
                                    text=text,
                                    pattern_type=inputs["pattern_type"],
                                    custom_pattern=inputs.get("custom_pattern")
                                )
                                logger.info(f"[TOOL_RESULT] {task_id}: extract_patterns - Found {result.get('count')} matches")
                            elif tool_name == "dataframe_ops":
                                params = inputs.get("params", {})
                                op = inputs.get("op")
                                
                                if not op:
                                    logger.error(f"[TOOL_EXEC] {task_id}: Missing 'op' parameter. Full inputs: {inputs}")
                                    raise ValueError(f"dataframe_ops requires 'op' parameter. Got inputs: {list(inputs.keys())}")
                                
                                logger.info(f"[TOOL_EXEC] {task_id}: dataframe_ops - op: {op}, params: {params}")
                                result = dataframe_ops(op, params)
                            elif tool_name == "make_plot":
                                result = make_plot(inputs["spec"])
                            elif tool_name == "zip_base64":
                                result = zip_base64(inputs["paths"])
                            elif tool_name == "call_llm":
                                logger.info(f"[TOOL_EXEC] {task_id}: call_llm")
                                prompt = inputs["prompt"]
                                for artifact_key in artifacts:
                                    artifact_value = str(artifacts[artifact_key])
                                    prompt = re.sub(r'\{\{' + re.escape(artifact_key) + r'\}\}', artifact_value, prompt)
                                    prompt = re.sub(r'\{' + re.escape(artifact_key) + r'\}', artifact_value, prompt)
                                system_prompt = inputs.get("system_prompt", "You are a helpful assistant.")
                                if "ONLY the" not in system_prompt and "only the" not in system_prompt:
                                    system_prompt += "\n\nIMPORTANT: Return ONLY the extracted value, no explanations or additional text."
                                result = await call_llm(prompt, system_prompt, inputs.get("max_tokens", 2000), inputs.get("temperature", 0))
                                logger.info(f"[TOOL_RESULT] {task_id}: call_llm - Response: {result[:200]}...")
                            elif tool_name == "answer_submit":
                                submit_url = inputs["url"]
                                submit_body = inputs["body"]
                                logger.info(f"[TOOL_EXEC] {task_id}: answer_submit - URL: {submit_url}")
                                
                                # URL validation and correction
                                if isinstance(submit_body, dict) and "url" in submit_body:
                                    body_url = str(submit_body["url"]).strip()
                                    origin_base = origin_url.split('?')[0]
                                    is_data_endpoint = "/data" in body_url or "/demo-scrape-data" in body_url
                                    is_wrong_url = body_url and body_url != origin_url and not body_url.startswith(origin_base)
                                    
                                    if (is_data_endpoint or is_wrong_url or body_url == "this page's URL"):
                                        logger.info(f"[TOOL_CORRECT] Replacing incorrect URL in body from {body_url} to {origin_url}")
                                        submit_body["url"] = origin_url
                                
                                # Variable substitution
                                if isinstance(submit_body, dict):
                                    processed_body = {}
                                    for key, value in submit_body.items():
                                        if isinstance(value, str):
                                            processed_value = value
                                            for artifact_key in artifacts:
                                                artifact_value = str(artifacts[artifact_key])
                                                processed_value = re.sub(r'\{\{' + re.escape(artifact_key) + r'\}\}', artifact_value, processed_value)
                                                processed_value = re.sub(r'\{' + re.escape(artifact_key) + r'\}', artifact_value, processed_value)
                                            if processed_value in artifacts:
                                                processed_value = str(artifacts[processed_value])
                                            if "your secret" in processed_value:
                                                processed_value = processed_value.replace("your secret", get_secret())
                                            if "this page's URL" in processed_value:
                                                processed_value = processed_value.replace("this page's URL", origin_url)
                                            if processed_value == "email":
                                                processed_value = email
                                            processed_body[key] = processed_value
                                        else:
                                            processed_body[key] = value
                                else:
                                    processed_body = submit_body
                                
                                logger.info(f"[TOOL_REQUEST_BODY] {task_id}: {json.dumps(processed_body, indent=2)[:300]}...")
                                result = await answer_submit(submit_url, processed_body)
                                logger.info(f"[TOOL_RESPONSE] {task_id}: Status {result.get('status_code')}")
                            
                            # New modular tools
                            elif tool_name == "scrape_with_javascript":
                                result = await ScrapingTools.scrape_with_javascript(inputs["url"], inputs.get("wait_for_selector"), inputs.get("timeout", 30000))
                                result = {"content": result}
                            elif tool_name == "fetch_from_api":
                                result = await ScrapingTools.fetch_from_api(inputs["url"], inputs.get("method", "GET"), inputs.get("headers"), inputs.get("body"), inputs.get("timeout", 30))
                            elif tool_name == "extract_html_text":
                                result = await ScrapingTools.extract_html_text(inputs["html"], inputs.get("selector"))
                                result = {"text": result}
                            elif tool_name == "clean_text":
                                result = CleansingTools.clean_text(inputs["text"], inputs.get("lowercase", False), inputs.get("remove_special", True), inputs.get("remove_whitespace", True))
                                result = {"cleaned_text": result}
                            elif tool_name == "extract_from_pdf":
                                result = CleansingTools.extract_from_pdf(inputs["pdf_path"], inputs.get("pages"))
                                result = {"text": result}
                            elif tool_name == "parse_csv_data":
                                df = CleansingTools.parse_csv_data(inputs["csv_content"], inputs.get("delimiter", ","))
                                result = {"dataframe": df, "shape": str(df.shape)}
                            elif tool_name == "parse_json_data":
                                result = CleansingTools.parse_json_data(inputs["json_content"])
                                result = {"data": result}
                            elif tool_name == "extract_structured_data":
                                result = CleansingTools.extract_structured_data(inputs["text"], inputs["pattern"])
                                result = {"matches": result}
                            elif tool_name == "transform_dataframe":
                                df = inputs["dataframe"]
                                result = ProcessingTools.transform_dataframe(df, inputs["operations"])
                                result = {"dataframe": result, "shape": str(result.shape)}
                            elif tool_name == "aggregate_data":
                                df = inputs["dataframe"]
                                result = ProcessingTools.aggregate_data(df, inputs["group_by"], inputs["aggregations"])
                                result = {"dataframe": result, "shape": str(result.shape)}
                            elif tool_name == "reshape_data":
                                df = inputs["dataframe"]
                                result = ProcessingTools.reshape_data(df, inputs["reshape_type"], **inputs.get("kwargs", {}))
                                result = {"dataframe": result, "shape": str(result.shape)}
                            elif tool_name == "transcribe_content":
                                result = ProcessingTools.transcribe_content(inputs["content"], inputs.get("format_type", "text"))
                                result = {"transcribed": result}
                            elif tool_name == "filter_data":
                                df = inputs["dataframe"]
                                result = AnalysisTools.filter_data(df, inputs["filters"])
                                result = {"dataframe": result, "shape": str(result.shape)}
                            elif tool_name == "sort_data":
                                df = inputs["dataframe"]
                                result = AnalysisTools.sort_data(df, inputs["sort_by"])
                                result = {"dataframe": result, "shape": str(result.shape)}
                            elif tool_name == "calculate_statistics":
                                from tool_executors import _dataframe_lock
                                
                                df_key = inputs["dataframe"]
                                
                                # Thread-safe registry read
                                with _dataframe_lock:
                                    if df_key not in dataframe_registry:
                                        raise ValueError(f"DataFrame '{df_key}' not found in registry. Available: {list(dataframe_registry.keys())}")
                                    df = dataframe_registry[df_key].copy()
                                
                                logger.info(f"[CALCULATE_STATS] DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
                                columns = inputs.get("columns", [])
                                # If no columns specified, use all numeric columns
                                if not columns:
                                    columns = df.select_dtypes(include=['number']).columns.tolist()
                                stats = inputs["stats"]
                                # Ensure stats is a list
                                if isinstance(stats, str):
                                    stats = [stats]
                                logger.info(f"[CALCULATE_STATS] Columns to analyze: {columns}, Stats: {stats}")
                                result = AnalysisTools.calculate_statistics(df, columns, stats)
                                logger.info(f"[CALCULATE_STATS] Result: {result}")
                                result = {"statistics": result}
                            elif tool_name == "apply_ml_model":
                                df = inputs["dataframe"]
                                result = AnalysisTools.apply_ml_model(df, inputs["model_type"], **inputs.get("kwargs", {}))
                                result = {"model_result": result}
                            elif tool_name == "train_linear_regression":
                                result = AnalysisTools.train_linear_regression(
                                    dataframe_key=inputs["dataframe_key"],
                                    feature_columns=inputs["feature_columns"],
                                    target_column=inputs["target_column"],
                                    predict_x=inputs.get("predict_x")
                                )
                                logger.info(f"[TOOL_RESULT] {task_id}: train_linear_regression - R²={result.get('r2_score', 'N/A')}, Prediction={result.get('prediction', 'N/A')}")
                            elif tool_name == "geospatial_analysis":
                                df = inputs.get("dataframe")
                                result = AnalysisTools.geospatial_analysis(df, inputs["analysis_type"], **inputs.get("kwargs", {}))
                                result = {"analysis_result": result}
                            elif tool_name == "create_chart":
                                # Infrastructure: Resolve dataframe_key from registry
                                from tool_executors import _dataframe_lock
                                
                                df_key = inputs["dataframe"]
                                
                                # Thread-safe registry read
                                with _dataframe_lock:
                                    if df_key not in dataframe_registry:
                                        raise ValueError(f"DataFrame '{df_key}' not found in registry. Available: {list(dataframe_registry.keys())}")
                                    df = dataframe_registry[df_key].copy()
                                
                                logger.info(f"[CREATE_CHART] DataFrame shape: {df.shape}, x_col: {inputs['x_col']}, y_col: {inputs['y_col']}")
                                
                                chart_path = VisualizationTools.create_chart(df, inputs["chart_type"], inputs["x_col"], inputs["y_col"], inputs.get("title", ""), inputs.get("output_path"))
                                
                                # Infrastructure: Count unique categories in x_col for quiz answers
                                unique_categories = df[inputs["x_col"]].nunique()
                                
                                logger.info(f"[CREATE_CHART] Chart created at {chart_path}, unique categories in {inputs['x_col']}: {unique_categories}")
                                result = {"chart_path": chart_path, "unique_categories": unique_categories}
                            elif tool_name == "create_interactive_chart":
                                # Infrastructure: Resolve dataframe_key from registry
                                from tool_executors import _dataframe_lock
                                
                                df_key = inputs["dataframe"]
                                
                                # Thread-safe registry read
                                with _dataframe_lock:
                                    if df_key not in dataframe_registry:
                                        raise ValueError(f"DataFrame '{df_key}' not found in registry. Available: {list(dataframe_registry.keys())}")
                                    df = dataframe_registry[df_key].copy()
                                
                                logger.info(f"[CREATE_INTERACTIVE_CHART] DataFrame shape: {df.shape}, x_col: {inputs['x_col']}, y_col: {inputs['y_col']}")
                                
                                chart_path = VisualizationTools.create_interactive_chart(df, inputs["chart_type"], inputs["x_col"], inputs["y_col"], inputs.get("title", ""), inputs.get("output_path"))
                                
                                # Infrastructure: Count unique categories in x_col for quiz answers
                                unique_categories = df[inputs["x_col"]].nunique()
                                
                                logger.info(f"[CREATE_INTERACTIVE_CHART] Chart created at {chart_path}, unique categories in {inputs['x_col']}: {unique_categories}")
                                result = {"chart_path": chart_path, "unique_categories": unique_categories}
                            elif tool_name == "generate_narrative":
                                df = inputs["dataframe"]
                                result = VisualizationTools.generate_narrative(df, inputs.get("summary_stats", {}))
                                result = {"narrative": result}
                            elif tool_name == "create_presentation_slide":
                                result = VisualizationTools.create_presentation_slide(inputs["title"], inputs["content"], inputs.get("output_path"))
                                result = {"slide_path": result}
                            else:
                                logger.warning(f"[TOOL_UNKNOWN] {task_id}: Unknown tool '{tool_name}' - skipping this task")
                                execution_log.append({
                                    "task_id": task_id,
                                    "status": "skipped",
                                    "tool": tool_name,
                                    "reason": f"Unknown tool: {tool_name}",
                                    "iteration": iteration
                                })
                                continue
                            
                            # Store artifacts
                            for produce in produces:
                                key = produce["key"]
                                if isinstance(result, dict) and key in result:
                                    artifacts[key] = result[key]
                                elif isinstance(result, (str, int, float, bool, list)):
                                    if tool_name == "call_llm" and isinstance(result, str):
                                        cleaned = result.strip()
                                        cleaned = re.sub(r'^.*?(?:is|are|be|found|extracted|as):\s*', '', cleaned, flags=re.IGNORECASE)
                                        cleaned = re.sub(r'^```.*?\n?', '', cleaned)
                                        cleaned = re.sub(r'\n?```$', '', cleaned)
                                        cleaned = re.sub(r'\s*[.!?]\s+.*$', '', cleaned)
                                        cleaned = cleaned.strip()
                                        artifacts[key] = cleaned if cleaned else result
                                    else:
                                        artifacts[key] = result
                                else:
                                    # For other dict results, store the entire dict
                                    if isinstance(result, dict):
                                        artifacts[key] = result
                                        logger.info(f"[ARTIFACT_STORAGE] Stored dict result for {key}: {str(result)[:200]}")
                                    else:
                                        # Special handling for render_js_page - extract rendered text
                                        if tool_name == "render_js_page" and isinstance(result, dict):
                                            rendered_divs = result.get("rendered_divs", [])
                                            if rendered_divs and len(rendered_divs) > 0:
                                                # Collect all rendered text from divs
                                                rendered_text = "\n".join([div.get("text", "") for div in rendered_divs if div.get("text", "").strip()])
                                                logger.info(f"[ARTIFACT_STORAGE] render_js_page: Extracted rendered text from {len(rendered_divs)} divs: {rendered_text[:100]}")
                                                artifacts[key] = rendered_text if rendered_text.strip() else result.get("text", str(result))
                                            else:
                                                # Fall back to body text if no rendered divs
                                                artifacts[key] = result.get("text", str(result))
                                        else:
                                            artifacts[key] = str(result)
                            
                            execution_log.append({
                                "task_id": task_id,
                                "status": "success",
                                "tool": tool_name,
                                "produces": [p["key"] for p in produces],
                                "iteration": iteration
                            })
                            
                        except Exception as e:
                            logger.error(f"Task {task_id} failed: {e}")
                            execution_log.append({
                                "task_id": task_id,
                                "status": "failed",
                                "tool": tool_name,
                                "error": str(e),
                                "iteration": iteration
                            })
                            raise
            
            # Format complex artifacts to extract meaningful values
            # This helps the completion checker understand if we have the answer
            if page_data and artifacts:
                page_instructions = page_data.get('text', '')
                formatted_artifacts = {}
                
                for key, value in artifacts.items():
                    # Skip if already extracted
                    extracted_key = f"extracted_{key}"
                    if extracted_key in artifacts:
                        continue
                    
                    # Only format the most recent artifacts (avoid re-processing old ones)
                    if any(task['iteration'] == iteration for task in execution_log if key in [p for produces in [t.get('produces', []) for t in all_tasks] for p in produces]):
                        formatted_value = await format_artifact(key, value, page_instructions)
                        if formatted_value != value:
                            # Store formatted version with new key
                            formatted_artifacts[extracted_key] = formatted_value
                            logger.info(f"[ARTIFACT_FORMAT] Created {extracted_key}: {str(formatted_value)[:100]}")
                
                # Add formatted artifacts to the collection
                artifacts.update(formatted_artifacts)
            
            logger.info("Checking if execution is complete...")
            
            if len(all_tasks) == 0:
                logger.info("Plan has 0 tasks - considered complete")
                break
            
            if page_data:
                completion_status = await check_plan_completion(
                    plan, 
                    artifacts, 
                    page_data,
                    last_executed_task=last_executed_task
                )
                logger.info(f"[PLAN_STATUS] {completion_status}")
                
                if completion_status.get("answer_ready", False):
                    logger.info("Answer is ready - stopping iterations")
                    break
                
                if completion_status.get("needs_more_tasks", False):
                    logger.info("Generating next batch of tasks...")
                    next_tasks = await generate_next_tasks(plan, artifacts, page_data, completion_status)
                    if next_tasks:
                        all_tasks = next_tasks
                        logger.info(f"Generated {len(all_tasks)} next tasks")
                    else:
                        logger.info("No more tasks generated - stopping iterations")
                        break
                else:
                    break
            else:
                break
        
        # Extract final answer
        final_answer = None
        final_spec = plan.get("final_answer_spec", {})
        page_text = page_data.get('text', '') if page_data else ''
        
        logger.info("Preparing final answer for submission")
        
        # Collect ALL candidate artifacts (not just one)
        candidate_artifacts = {}
        
        # Gather all potentially useful artifacts
        for key in artifacts.keys():
            value = artifacts[key]
            
            # Skip internal tracking artifacts
            if key.startswith('_'):
                continue
            
            # Include statistics results
            if isinstance(value, dict) and 'statistics' in value:
                candidate_artifacts[key] = value
                continue
            
            # Include successful transcription artifacts
            if isinstance(value, dict) and value.get('success') and 'text' in value:
                if 'audio_path' in value or 'method' in value or key.startswith('transcribe'):
                    candidate_artifacts[key] = value
                    continue
            
            # Include pattern extraction results
            if isinstance(value, dict) and 'pattern_type' in value and 'count' in value:
                candidate_artifacts[key] = value
                continue
            
            # Include vision results
            if isinstance(value, dict) and ('vision_result' in value or 'ocr_text' in value or 'result' in value):
                # Check if it's from a vision/image tool
                if 'vision' in key.lower() or 'image' in key.lower() or 'analyze_image' in key:
                    candidate_artifacts[key] = value
                    continue
            
            # Include dataframe operation results
            if isinstance(value, dict) and 'result' in value and 'type' in value:
                candidate_artifacts[key] = value
                continue
            
            # Include chart/plot paths
            if isinstance(value, dict) and ('chart_path' in value or 'plot_path' in value):
                candidate_artifacts[key] = value
                continue
            
            # Include reasonable string artifacts (not HTML/scripts)
            if isinstance(value, str) and len(value) < 500 and len(value) > 2:
                if not ('<' in value or 'script' in value.lower()):
                    prefixes = ('extracted_', 'rendered_', 'scraped_', 'secret_', 'result_', 'answer_', 'transcription_')
                    if any(key.startswith(p) for p in prefixes):
                        candidate_artifacts[key] = value
                        continue
            
            # Include dict artifacts with 'text' field (e.g., rendered pages)
            if isinstance(value, dict) and 'text' in value:
                text_content = value.get('text', '')
                if isinstance(text_content, str) and len(text_content.strip()) > 5:
                    if not ('<' in text_content or 'script' in text_content.lower()):
                        candidate_artifacts[key] = value
                        continue
        
        # If we have multiple candidates, ask LLM to choose the correct one
        if len(candidate_artifacts) > 1:
            logger.info(f"[ARTIFACT_SELECTION] Found {len(candidate_artifacts)} candidate artifacts - asking LLM to select")
            
            # Prepare artifact summaries for LLM
            artifact_summaries = {}
            for key, value in candidate_artifacts.items():
                if isinstance(value, dict):
                    # For dicts, show structure
                    if 'text' in value:
                        artifact_summaries[key] = f"Dict with 'text': {str(value['text'])[:100]}..."
                    elif 'statistics' in value:
                        artifact_summaries[key] = f"Statistics result: {value}"
                    elif 'count' in value and 'pattern_type' in value:
                        artifact_summaries[key] = f"Pattern extraction: {value['pattern_type']} (count={value['count']})"
                    elif 'success' in value and 'audio_path' in value:
                        artifact_summaries[key] = f"Transcription: {value.get('text', '')}"
                    else:
                        artifact_summaries[key] = f"Dict: {str(value)[:100]}..."
                else:
                    artifact_summaries[key] = f"String: {str(value)[:100]}..."
            
            # Ask LLM which artifact contains the answer
            selection_prompt = f"""You have multiple artifacts that could contain the answer to this question:

Question: {page_text}

Available artifacts:
{json.dumps(artifact_summaries, indent=2)}

Which artifact key contains the actual answer to the question? Respond with ONLY the artifact key name, nothing else."""

            try:
                selection_response = await call_llm(
                    prompt=selection_prompt,
                    system_prompt="You are selecting the correct artifact. Respond with ONLY the artifact key name."
                )
                selected_key = selection_response.strip()
                
                # Validate the selection
                if selected_key in candidate_artifacts:
                    logger.info(f"[ARTIFACT_SELECTION] LLM selected: {selected_key}")
                    final_answer = candidate_artifacts[selected_key]
                else:
                    # Fallback: use most recent
                    logger.warning(f"[ARTIFACT_SELECTION] LLM selected invalid key '{selected_key}', using most recent")
                    final_answer = list(candidate_artifacts.values())[-1]
                    selected_key = list(candidate_artifacts.keys())[-1]
            except Exception as e:
                logger.error(f"[ARTIFACT_SELECTION] LLM selection failed: {e}, using most recent")
                final_answer = list(candidate_artifacts.values())[-1]
                selected_key = list(candidate_artifacts.keys())[-1]
        elif len(candidate_artifacts) == 1:
            # Only one candidate - use it
            final_answer = list(candidate_artifacts.values())[0]
            selected_key = list(candidate_artifacts.keys())[0]
            logger.info(f"[ARTIFACT_SELECTION] Single candidate: {selected_key}")
        else:
            # No candidates found - check if plan specified an artifact
            from_key = final_spec.get("from", "")
            if from_key:
                logger.warning(f"[ARTIFACT_SELECTION] No candidates found, trying plan-specified '{from_key}'")
                cleaned_key = from_key.strip()
                cleaned_key = re.sub(r"^static value\s*['\"]?|['\"]?$", "", cleaned_key)
                cleaned_key = re.sub(r"^['\"]|['\"]$", "", cleaned_key).strip()
                
                # Infrastructure: Handle nested field access with dot notation
                # Example: "chart_result.unique_categories" → artifacts["chart_result"]["unique_categories"]
                if '.' in cleaned_key:
                    parts = cleaned_key.split('.')
                    base_key = parts[0]
                    if base_key in artifacts:
                        final_answer = artifacts[base_key]
                        # Navigate nested fields
                        for field in parts[1:]:
                            if isinstance(final_answer, dict) and field in final_answer:
                                final_answer = final_answer[field]
                                logger.info(f"[ARTIFACT_NESTED] Extracted field '{field}' from artifact '{base_key}'")
                            else:
                                logger.warning(f"[ARTIFACT_NESTED] Field '{field}' not found in artifact '{base_key}'")
                                final_answer = None
                                break
                        selected_key = cleaned_key if final_answer is not None else "none"
                    else:
                        logger.warning(f"[ARTIFACT_NESTED] Base artifact '{base_key}' not found")
                        final_answer = cleaned_key  # Use as literal
                        selected_key = "literal"
                elif from_key in artifacts:
                    final_answer = artifacts[from_key]
                    selected_key = from_key
                elif cleaned_key in artifacts:
                    final_answer = artifacts[cleaned_key]
                    selected_key = cleaned_key
                else:
                    logger.warning(f"[ARTIFACT_SELECTION] '{from_key}' not found, using as literal")
                    final_answer = cleaned_key
                    selected_key = "literal"
            else:
                logger.error("[ARTIFACT_SELECTION] No candidates and no plan-specified artifact!")
                final_answer = None
                selected_key = "none"
        
        # Handle dict objects and string representations of dicts
        if isinstance(final_answer, dict):
            # Handle statistics results: {'statistics': {'column': {'sum': value}}}
            if 'statistics' in final_answer:
                logger.info(f"[ARTIFACT_EXTRACTION] Detected statistics result: {final_answer}")
                stats_dict = final_answer['statistics']
                # Extract the actual values from nested structure
                if isinstance(stats_dict, dict):
                    # Get first column's statistics
                    first_col = next(iter(stats_dict.keys()))
                    col_stats = stats_dict[first_col]
                    if isinstance(col_stats, dict) and len(col_stats) > 0:
                        # If only one stat, return that value
                        if len(col_stats) == 1:
                            stat_value = next(iter(col_stats.values()))
                            # Handle numpy types
                            if hasattr(stat_value, 'item'):
                                stat_value = stat_value.item()
                            logger.info(f"[ARTIFACT_EXTRACTION] Extracted single statistic value: {stat_value}")
                            final_answer = stat_value
                        else:
                            # Multiple stats - prefer sum, then mean, then first
                            if 'sum' in col_stats:
                                final_answer = col_stats['sum']
                                if hasattr(final_answer, 'item'):
                                    final_answer = final_answer.item()
                                logger.info(f"[ARTIFACT_EXTRACTION] Extracted 'sum': {final_answer}")
                            elif 'mean' in col_stats:
                                final_answer = col_stats['mean']
                                if hasattr(final_answer, 'item'):
                                    final_answer = final_answer.item()
                                logger.info(f"[ARTIFACT_EXTRACTION] Extracted 'mean': {final_answer}")
                            else:
                                final_answer = next(iter(col_stats.values()))
            
            # Handle extract_patterns result format: {'matches': [...], 'count': N, 'pattern_type': 'email'}
            if isinstance(final_answer, dict) and 'pattern_type' in final_answer and 'count' in final_answer:
                logger.info(f"[ARTIFACT_EXTRACTION] Detected extract_patterns result: {final_answer}")
                # Extract just the count for "how many" questions
                final_answer = final_answer['count']
                logger.info(f"[ARTIFACT_EXTRACTION] Extracted count: {final_answer}")
            
            # Infrastructure: Handle chart result format: {'chart_path': 'path', 'unique_categories': N}
            # For "how many categories" questions, extract the count
            if isinstance(final_answer, dict) and 'chart_path' in final_answer and 'unique_categories' in final_answer:
                logger.info(f"[ARTIFACT_EXTRACTION] Detected chart result: {final_answer}")
                # Check if question asks for category count
                if page_text and ('how many categor' in page_text.lower() or 'number of categor' in page_text.lower()):
                    final_answer = final_answer['unique_categories']
                    logger.info(f"[ARTIFACT_EXTRACTION] Extracted unique_categories for count question: {final_answer}")
            
            # Handle train_linear_regression result format: {'prediction': value, 'coefficients': [...], ...}
            if isinstance(final_answer, dict) and 'prediction' in final_answer and 'model_type' in final_answer:
                logger.info(f"[ARTIFACT_EXTRACTION] Detected train_linear_regression result: {final_answer}")
                # Extract just the prediction value
                final_answer = final_answer['prediction']
                logger.info(f"[ARTIFACT_EXTRACTION] Extracted prediction: {final_answer}")
            
            # Handle vision/OCR result format: {'result': 'text'}
            if isinstance(final_answer, dict) and 'result' in final_answer and len(final_answer) == 1:
                logger.info(f"[ARTIFACT_EXTRACTION] Detected vision/OCR result: {final_answer}")
                final_answer = final_answer['result']
                logger.info(f"[ARTIFACT_EXTRACTION] Extracted vision result: {final_answer}")
            
            # Handle dataframe_ops result format: {'result': value, 'type': 'typename'}
            elif isinstance(final_answer, dict) and 'result' in final_answer and 'type' in final_answer:
                result_value = final_answer['result']
                # Handle numpy types
                if hasattr(result_value, 'item'):
                    result_value = result_value.item()
                logger.info(f"[ARTIFACT_EXTRACTION] Extracted result value: {result_value}")
                final_answer = result_value
            # Actual dict object with 'text' key
            elif isinstance(final_answer, dict) and 'text' in final_answer:
                logger.info(f"[ARTIFACT_EXTRACTION] Extracting 'text' from dict object")
                final_answer = final_answer['text']
                logger.info(f"[ARTIFACT_EXTRACTION] Extracted: {str(final_answer)[:100]}...")
        
        # Extract secret code/value from rendered text patterns
        if isinstance(final_answer, str):
            # Pattern: "The secret code is: 9876" or "Secret code is 1371" etc.
            secret_patterns = [
                r'(?:secret code|code|answer|value)\s*(?:is|:)\s*[:\-]?\s*(\d+)',
                r'(?:secret code|code|answer|value)\s*[:\-]\s*[:\-]?\s*(\S+)',
            ]
            for pattern in secret_patterns:
                match = re.search(pattern, final_answer, re.IGNORECASE)
                if match:
                    extracted_value = match.group(1)
                    logger.info(f"[ARTIFACT_EXTRACTION] Detected secret code pattern, extracting: {extracted_value}")
                    final_answer = extracted_value
                    break
        
        elif isinstance(final_answer, str) and (final_answer.startswith("{'") or final_answer.startswith('{"')):
            # String representation of a dict - try to extract 'text'
            logger.info(f"[ARTIFACT_EXTRACTION] Detected string dict representation, extracting 'text'...")
            try:
                import ast
                dict_obj = ast.literal_eval(final_answer)
                if isinstance(dict_obj, dict) and 'text' in dict_obj:
                    final_answer = dict_obj['text']
                    logger.info(f"[ARTIFACT_EXTRACTION] Extracted text from dict string: {str(final_answer)[:100]}...")
            except (ValueError, SyntaxError):
                logger.info(f"[ARTIFACT_EXTRACTION] Could not parse dict string, using as-is")
        
        # Parse numbers or codes from natural language responses
        if isinstance(final_answer, str):
            # Check for pattern "Secret code is X" or similar
            # Try specific patterns first
            patterns = [
                r'Secret code is (\d+)',  # "Secret code is 1371"
                r'code is (\d+)',  # "code is 1371"
                r'secret is (\d+)',  # "secret is 1371"
                r'answer is (\d+)',  # "answer is 1371"
                r'value is (\d+)',  # "value is 1371"
            ]
            
            matched = False
            for pattern in patterns:
                match = re.search(pattern, final_answer, re.IGNORECASE)
                if match:
                    extracted_value = match.group(1)
                    logger.info(f"[ANSWER_PARSING] Extracted '{extracted_value}' from: {final_answer[:100]}")
                    final_answer = extracted_value
                    matched = True
                    break
            
            # If no pattern matched but answer looks like natural language with numbers, extract first number
            if not matched and len(final_answer) > 20:
                numbers = re.findall(r'\b\d+\b', final_answer)
                if numbers:
                    # If text contains phrases like "and not", take first number as likely answer
                    if "and not" in final_answer.lower() or "not" in final_answer.lower():
                        logger.info(f"[ANSWER_PARSING] Found 'not' pattern, extracting first number: {numbers[0]} from: {final_answer[:100]}")
                        final_answer = numbers[0]
        
        if not final_answer and len(all_tasks) == 0:
            logger.info("Plan has no tasks and no artifact reference, providing default answer")
            answer_type = final_spec.get("type", "string")
            if answer_type == "boolean":
                final_answer = "true"
            elif answer_type == "number":
                final_answer = "0"
            elif answer_type == "json":
                final_answer = "{}"
            else:
                final_answer = "completed"
        
        # Submit answer
        logger.info(f"Final answer ready for submission: {final_answer}")
        
        # Convert numpy types to native Python types for JSON serialization
        if hasattr(final_answer, 'item'):
            final_answer = final_answer.item()
        
        if quiz_attempt:
            quiz_attempt.answer = final_answer
        
        # Build submission payload
        request_body_spec = plan.get("request_body", {})
        submit_url = plan.get("submit_url", "")
        
        # Build submission body using old executor logic
        submit_body = {}
        
        if request_body_spec:
            email_key = request_body_spec.get("email_key", "email")
            submit_body[email_key] = email
            
            secret_key = request_body_spec.get("secret_key", "secret")
            submit_body[secret_key] = get_secret()
            
            url_key = request_body_spec.get("url_value", "url")
            if url_key:
                submit_body[url_key] = origin_url
            
            answer_key = request_body_spec.get("answer_key", "answer")
            submit_body[answer_key] = final_answer
        
        logger.info(f"Submitting answer to {submit_url}")
        logger.info(f"Request body: {json.dumps(submit_body, indent=2)}")
        
        submission_result = await answer_submit(submit_url, submit_body)
        
        if quiz_attempt:
            quiz_attempt.submission_response = submission_result.get("response", {})
            quiz_attempt.correct = submission_result.get("response", {}).get("correct", False)
        
        return {
            "success": True,
            "final_answer": final_answer,
            "submission_result": submission_result,
            "artifacts": artifacts,
            "execution_log": execution_log,
            "iterations_used": iteration
        }
        
    except Exception as e:
        logger.error(f"Plan execution failed: {e}")
        import traceback
        traceback.print_exc()
        if quiz_attempt:
            quiz_attempt.error = str(e)
        return {
            "success": False,
            "error": str(e),
            "artifacts": artifacts if 'artifacts' in locals() else {},
            "execution_log": execution_log if 'execution_log' in locals() else []
        }


async def run_pipeline(email: str, url: str) -> Dict[str, Any]:
    """
    Run the complete quiz-solving pipeline for a single URL
    Handles multiple sequential quizzes and retry logic
    """
    try:
        quiz_runs = {}
        quiz_chain = []
        current_url = url
        
        while True:
            logger.info(f"\n{'='*60}")
            logger.info(f"SOLVING QUIZ: {current_url}")
            logger.info(f"{'='*60}\n")
            
            # Initialize QuizRun for this URL
            if current_url not in quiz_runs:
                quiz_runs[current_url] = QuizRun(quiz_url=current_url)
            
            quiz_run = quiz_runs[current_url]
            
            # Retry loop for incorrect answers
            while True:
                # Create new attempt
                quiz_attempt = quiz_run.start_attempt()
                logger.info(f"[QUIZ_RUN] Starting attempt {quiz_attempt.attempt_number} for {current_url}")
                
                # Render page
                logger.info(f"[QUIZ_RUN] Rendering quiz page")
                page_data = await render_page(current_url)
                logger.info(f"[QUIZ_RUN] Page rendered - text length: {len(page_data.get('text', ''))}")
                
                # Generate execution plan
                logger.info(f"[QUIZ_RUN] Generating execution plan")
                plan_response = await call_llm_for_plan(page_data, quiz_run.attempts[:-1] if len(quiz_run.attempts) > 1 else None)
                plan_json = extract_json_from_markdown(plan_response)
                
                try:
                    plan_obj = json.loads(plan_json)
                    logger.info(f"[QUIZ_RUN] Plan parsed successfully")
                except json.JSONDecodeError as e:
                    logger.error(f"[QUIZ_RUN] Failed to parse plan JSON: {e}")
                    logger.error(f"[QUIZ_RUN] Raw plan response: {plan_response}")
                    quiz_attempt.error = f"Failed to parse plan: {e}"
                    quiz_attempt.finish()
                    quiz_run.finish_current_attempt()
                    return {
                        "success": False,
                        "error": f"Failed to parse execution plan: {e}",
                        "quiz_runs": {url: qr.to_dict() for url, qr in quiz_runs.items()}
                    }
                
                # Add previous attempts context if this is a retry
                previous_attempts_context = ""
                if len(quiz_run.attempts) > 1:
                    previous_attempts_context = "\n\nPREVIOUS ATTEMPTS:"
                    for prev_attempt in quiz_run.attempts[:-1]:  # Exclude current attempt
                        previous_attempts_context += f"\nAttempt {prev_attempt.attempt_number}:\n"
                        previous_attempts_context += f"  - Answer: {prev_attempt.answer}\n"
                        previous_attempts_context += f"  - Result: {'Correct' if prev_attempt.correct else 'Incorrect'}\n"
                        if prev_attempt.error:
                            previous_attempts_context += f"  - Error: {prev_attempt.error}\n"
                    logger.info(f"[QUIZ_RUN] Context from {len(quiz_run.attempts)-1} previous attempts prepared")
                
                # Execute plan with attempt tracking
                logger.info(f"[QUIZ_RUN] Executing plan for attempt {quiz_attempt.attempt_number}")
                execution_result = await execute_plan(plan_obj, email, current_url, page_data, quiz_attempt)
                logger.info(f"[QUIZ_EXECUTION] Execution result: success={execution_result.get('success')}, final_answer={execution_result.get('final_answer')}")
                
                quiz_attempt.finish()
                
                if not execution_result.get("success"):
                    logger.error(f"[QUIZ_RUN] Execution failed at attempt {quiz_attempt.attempt_number}")
                    quiz_run.finish_current_attempt()
                    return {
                        "success": False,
                        "error": execution_result.get("error", "Unknown error"),
                        "quiz_runs": {url: qr.to_dict() for url, qr in quiz_runs.items()}
                    }
                
                # Check submission response
                submission_response = execution_result.get("submission_result", {})
                response_data = submission_response.get("response", {}) if submission_response else {}
                
                logger.info(f"[QUIZ_RESPONSE] Submission response: {submission_response}")
                logger.info(f"[QUIZ_RESPONSE] Full response data: {json.dumps(response_data, indent=2)}")
                
                # Check if answer was correct
                is_correct = response_data.get("correct", False)
                logger.info(f"[QUIZ_RESPONSE] Answer correct: {is_correct}")
                
                # Store attempt results
                quiz_attempt.submission_response = response_data
                quiz_attempt.correct = is_correct
                quiz_run.finish_current_attempt()
                
                if is_correct:
                    logger.info(f"[QUIZ_SUCCESS] Quiz solved successfully on attempt {quiz_attempt.attempt_number}")
                    
                    # Check if there's a next quiz URL
                    if "url" in response_data and response_data["url"]:
                        next_url = response_data["url"]
                        logger.info(f"[QUIZ_CHAIN] Next quiz found at: {next_url}")
                        
                        # Add successful run to chain
                        quiz_chain.append({
                            "quiz_url": current_url,
                            "quiz_run": quiz_run.to_dict()
                        })
                        
                        # Move to next quiz (attempt counter will reset with new QuizRun)
                        current_url = next_url
                        break  # Break inner retry loop to start new quiz
                    else:
                        logger.info("[QUIZ_COMPLETE] No next quiz URL. Quiz chain complete!")
                        quiz_chain.append({
                            "quiz_url": current_url,
                            "quiz_run": quiz_run.to_dict()
                        })
                        
                        # Log cache statistics
                        quiz_cache.log_stats()
                        
                        # Log completion check statistics
                        log_completion_stats()
                        break
                else:
                    # Answer was incorrect, check if we can retry
                    elapsed_time = quiz_run.elapsed_time_since_first()
                    avg_time = quiz_run.average_time_per_attempt()
                    max_retry_time = 180  # 3 minutes in seconds
                    remaining_time = max_retry_time - elapsed_time
                    
                    logger.info(f"[QUIZ_RETRY] Answer was incorrect. Elapsed: {elapsed_time:.1f}s, Avg per attempt: {avg_time:.1f}s, Remaining: {remaining_time:.1f}s, Attempts: {quiz_attempt.attempt_number}")
                    
                    # Smart retry: check if average time per attempt < remaining time
                    if quiz_run.can_retry_smart(max_retry_time):
                        logger.info(f"[QUIZ_RETRY] Smart retry enabled - avg time ({avg_time:.1f}s) < remaining time ({remaining_time:.1f}s). Attempting again...")
                        # Loop continues to next attempt
                        continue
                    else:
                        logger.error(f"[QUIZ_FAILED] Not enough time for retry. Avg time: {avg_time:.1f}s >= Remaining: {remaining_time:.1f}s after {quiz_attempt.attempt_number} attempts.")
                        quiz_chain.append({
                            "quiz_url": current_url,
                            "quiz_run": quiz_run.to_dict(),
                            "failed": True,
                            "reason": f"Not enough time for retry (avg: {avg_time:.1f}s, remaining: {remaining_time:.1f}s)"
                        })
                        break
            
            # Check if we should continue to next quiz (success case)
            if is_correct and "url" in response_data and response_data["url"]:
                # Continue outer loop with new URL (already set above)
                continue
            else:
                # No next quiz or quiz failed - exit pipeline
                break
        
        logger.info(f"[PIPELINE_COMPLETE] Quiz chain complete. Solved {len(quiz_chain)} quizzes")
        
        # Log final cache statistics
        quiz_cache.log_stats()
        
        # Log completion check statistics
        log_completion_stats()
        
        return {
            "success": True,
            "quiz_chain": quiz_chain,
            "quiz_runs": {url: qr.to_dict() for url, qr in quiz_runs.items()},
            "total_quizzes_solved": len(quiz_chain),
            "final_result": quiz_chain[-1] if quiz_chain else None
        }
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
