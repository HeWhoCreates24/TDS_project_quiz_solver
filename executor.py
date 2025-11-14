"""
Tool execution functions and plan execution engine
"""
import os
import time
import json
import logging
import re
import asyncio
import base64
import io
import zipfile
from typing import Any, Dict, List, Optional
from tempfile import NamedTemporaryFile
from playwright.async_api import async_playwright
import httpx
import pandas as pd
import matplotlib.pyplot as plt
from tools import ToolRegistry, ScrapingTools, CleansingTools, ProcessingTools, AnalysisTools, VisualizationTools
from models import QuizAttempt

logger = logging.getLogger(__name__)

# Global registry for dataframes
dataframe_registry = {}
SECRET = os.getenv("SECRET")

# ===== BASIC TOOL FUNCTIONS =====

async def render_page(url: str) -> Dict[str, Any]:
    """Render page with Playwright and extract content"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle", timeout=45000)
        content = await page.content()
        text = await page.evaluate("() => document.body.innerText")
        links = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
        pres = await page.eval_on_selector_all("pre,code", "els => els.map(e => e.innerText)")
        await browser.close()
        return {"html": content, "text": text, "links": links, "code_blocks": pres}

async def fetch_text(url: str) -> Dict[str, Any]:
    """Fetch text content from URL"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30)
            response.raise_for_status()
            return {
                "text": response.text,
                "headers": dict(response.headers),
                "status_code": response.status_code
            }
    except Exception as e:
        logger.error(f"Error fetching text from {url}: {e}")
        raise

async def download_file(url: str) -> Dict[str, Any]:
    """Download file and return metadata"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=60)
            response.raise_for_status()
            
            with NamedTemporaryFile(delete=False, suffix=".download") as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            
            return {
                "path": tmp_path,
                "content_type": response.headers.get("content-type", "unknown"),
                "size": len(response.content),
                "filename": url.split("/")[-1]
            }
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {e}")
        raise

def parse_csv(path: str) -> Dict[str, Any]:
    """Parse CSV file"""
    try:
        df = pd.read_csv(path)
        dataframe_registry[f"df_{len(dataframe_registry)}"] = df
        return {
            "dataframe_key": f"df_{len(dataframe_registry)-1}",
            "shape": df.shape,
            "columns": list(df.columns),
            "sample": df.head().to_dict()
        }
    except Exception as e:
        logger.error(f"Error parsing CSV {path}: {e}")
        raise

def parse_excel(path: str) -> Dict[str, Any]:
    """Parse Excel file"""
    try:
        df = pd.read_excel(path)
        dataframe_registry[f"df_{len(dataframe_registry)}"] = df
        return {
            "dataframe_key": f"df_{len(dataframe_registry)-1}",
            "shape": df.shape,
            "columns": list(df.columns),
            "sample": df.head().to_dict()
        }
    except Exception as e:
        logger.error(f"Error parsing Excel {path}: {e}")
        raise

def parse_json_file(path: str) -> Dict[str, Any]:
    """Parse JSON file"""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return {"data": data, "type": type(data).__name__}
    except Exception as e:
        logger.error(f"Error parsing JSON {path}: {e}")
        raise

def parse_html_tables(html_content: str) -> Dict[str, Any]:
    """Parse HTML tables"""
    try:
        tables = pd.read_html(html_content)
        result = {}
        for i, table in enumerate(tables):
            key = f"table_{i}"
            dataframe_registry[key] = table
            result[key] = {
                "shape": table.shape,
                "columns": list(table.columns),
                "sample": table.head().to_dict()
            }
        return {"tables": result}
    except Exception as e:
        logger.error(f"Error parsing HTML tables: {e}")
        raise

def parse_pdf_tables(path: str, pages: str = "all") -> Dict[str, Any]:
    """Parse PDF tables (placeholder)"""
    try:
        return {
            "warning": "PDF parsing requires pdfplumber installation",
            "path": path,
            "pages": pages
        }
    except Exception as e:
        logger.error(f"Error parsing PDF {path}: {e}")
        raise

def dataframe_ops(op: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform DataFrame operations"""
    try:
        df_key = params.get("dataframe_key")
        if df_key not in dataframe_registry:
            raise ValueError(f"DataFrame {df_key} not found in registry")
        
        df = dataframe_registry[df_key]
        result = None
        
        if op == "select":
            columns = params.get("columns", [])
            result = df[columns] if columns else df
        elif op == "filter":
            condition = params.get("condition")
            result = df.query(condition) if condition else df
        elif op == "sum":
            column = params.get("column")
            result = df[column].sum() if column else df.sum()
        elif op == "mean":
            column = params.get("column")
            result = df[column].mean() if column else df.mean()
        elif op == "groupby":
            by = params.get("by")
            agg = params.get("aggregation", "count")
            result = df.groupby(by).agg(agg)
        elif op == "count":
            result = len(df)
        else:
            raise ValueError(f"Unknown operation: {op}")
        
        if isinstance(result, pd.DataFrame):
            new_key = f"df_{len(dataframe_registry)}"
            dataframe_registry[new_key] = result
            return {
                "dataframe_key": new_key,
                "result": "DataFrame operation completed",
                "shape": result.shape
            }
        else:
            return {"result": result, "type": type(result).__name__}
            
    except Exception as e:
        logger.error(f"Error in dataframe operation {op}: {e}")
        raise

def make_plot(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Create plot and return base64 data URI"""
    try:
        plt.figure(figsize=spec.get("figsize", (10, 6)))
        
        plot_type = spec.get("type", "line")
        data_key = spec.get("dataframe_key")
        
        if data_key in dataframe_registry:
            df = dataframe_registry[data_key]
            x = spec.get("x")
            y = spec.get("y")
            
            if plot_type == "line":
                plt.plot(df[x] if x else df.index, df[y] if y else df.values)
            elif plot_type == "bar":
                plt.bar(df[x] if x else df.index, df[y] if y else df.values)
            elif plot_type == "scatter":
                plt.scatter(df[x], df[y])
            elif plot_type == "histogram":
                plt.hist(df[y] if y else df.values.flatten())
        else:
            data = spec.get("data", [])
            plt.plot(data)
        
        plt.title(spec.get("title", "Plot"))
        plt.xlabel(spec.get("xlabel", ""))
        plt.ylabel(spec.get("ylabel", ""))
        
        if spec.get("grid"):
            plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return {"base64_uri": f"data:image/png;base64,{img_base64}"}
        
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        raise

def zip_base64(paths: List[str]) -> Dict[str, Any]:
    """Zip files and return base64 data URI"""
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zip_file:
            for path in paths:
                zip_file.write(path, os.path.basename(path))
        
        buf.seek(0)
        zip_base64_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return {"base64_uri": f"data:application/zip;base64,{zip_base64_str}"}
        
    except Exception as e:
        logger.error(f"Error creating zip: {e}")
        raise

async def answer_submit(url: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Submit answer to quiz endpoint"""
    try:
        logger.info(f"[API_REQUEST] POST {url}")
        logger.info(f"[API_REQUEST_BODY] {json.dumps(body, indent=2)[:500]}...")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=body, timeout=30)
            response.raise_for_status()
            
            response_data = response.json() if response.content else {}
            logger.info(f"[API_RESPONSE] Status: {response.status_code}")
            logger.info(f"[API_RESPONSE_BODY] {json.dumps(response_data, indent=2)[:500]}...")
            
            return {
                "status_code": response.status_code,
                "response": response_data,
                "headers": dict(response.headers)
            }
    except Exception as e:
        logger.error(f"[API_ERROR] Error submitting answer to {url}: {e}")
        if 'response' in locals():
            logger.error(f"[API_ERROR_RESPONSE] {response.text[:500]}")
        logger.error(f"[API_ERROR_BODY] {json.dumps(body, indent=2)[:500]}")
        print(f"Submission body: {body}")
        raise

async def call_llm(prompt: str, system_prompt: str = None, max_tokens: int = 2000, temperature: float = 0) -> str:
    """Call LLM with given prompt"""
    if system_prompt is None:
        system_prompt = "You are a helpful assistant."
    
    OPEN_AI_BASE_URL = "https://aipipe.org/openai/v1/chat/completions"
    API_KEY = os.getenv("API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    json_data = {
        "model": "gpt-4o-2024-08-06",
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

async def call_llm_for_plan(page_data: Dict[str, Any]) -> str:
    """Generate execution plan using LLM"""
    system_prompt = """
        You are an execution planner for an automated quiz-solving agent. Your job is to read a rendered quiz page 
        (HTML, visible text, links, code/pre blocks), infer the required steps, and emit a machine-executable plan as strict JSON. 
        Do not perform the tasks yourself. Only plan and specify tools and inputs.

        Output JSON Schema must match the specified format.
        Return ONLY valid JSON, no extra prose.
        """
    
    prompt = f"""
        Rendered quiz page extraction:
        html: {page_data['html']}
        text: {page_data['text']}
        code_blocks: {page_data['code_blocks']}
        links: {page_data['links']}

        Your job:
        1. Identify the submit URL
        2. Identify the URL value that must be echoed in the request body
        3. Identify the instruction for the answer and required answer type
        4. Plan the minimal tasks to produce the answer
        5. Output the final JSON plan only
        """
    
    OPEN_AI_BASE_URL = "https://aipipe.org/openai/v1/chat/completions"
    API_KEY = os.getenv("API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    json_data = {
        "model": "gpt-4o-2024-08-06",
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
        return plan_text

async def check_plan_completion(plan_obj: Dict[str, Any], artifacts: Dict[str, Any], page_data: Dict[str, Any]) -> Dict[str, Any]:
    """Check if plan execution is complete"""
    system_prompt = """
        You are a task planner evaluating execution progress. Determine if the answer is ready to submit or if more tasks are needed.
        
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
        if isinstance(value, (str, int, float, bool, list, dict)):
            artifacts_summary[key] = value
        else:
            artifacts_summary[key] = str(value)[:500]
    
    prompt = f"""
        Current Plan:
        {json.dumps(plan_obj.get('final_answer_spec', {}), indent=2)}
        
        Produced Artifacts:
        {json.dumps(artifacts_summary, indent=2)}
        
        Quiz Instructions:
        {page_data.get('text', 'N/A')[:1000]}
        
        Based on the artifacts we have so far, can we extract the final answer?
        If not, what additional data or tasks do we need?
        """
    
    response_text = await call_llm(prompt, system_prompt, 1000, 0)
    
    try:
        cleaned_response = response_text.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response)
        cleaned_response = re.sub(r'\n?```$', '', cleaned_response)
        result = json.loads(cleaned_response)
    except json.JSONDecodeError:
        result = {
            "answer_ready": False,
            "needs_more_tasks": True,
            "reason": response_text,
            "recommended_next_action": response_text
        }
    
    return result

async def generate_next_tasks(plan_obj: Dict[str, Any], artifacts: Dict[str, Any], page_data: Dict[str, Any], completion_status: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate next batch of tasks"""
    system_prompt = """
        You are an execution planner generating the NEXT BATCH of tasks needed to proceed.
        Return a JSON array of task objects only. Each task should have:
        { "id": "string", "tool_name": "string", "inputs": {}, "produces": [{"key": "string", "type": "string"}], "notes": "string" }
        Return ONLY valid JSON array, no other text.
        """
    
    artifacts_summary = {}
    for key, value in artifacts.items():
        if isinstance(value, (str, int, float, bool, list, dict)):
            artifacts_summary[key] = value
        else:
            artifacts_summary[key] = str(value)[:500]
    
    prompt = f"""
        Current Artifacts:
        {json.dumps(artifacts_summary, indent=2)}
        
        Completion Analysis:
        {json.dumps(completion_status, indent=2)}
        
        Quiz Page Info:
        Text: {page_data.get('text', 'N/A')[:1000]}
        Links: {page_data.get('links', [])}
        
        Generate the NEXT BATCH of tasks to gather the missing information.
        """
    
    response_text = await call_llm(prompt, system_prompt, 2000, 0)
    
    try:
        cleaned_response = response_text.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response)
        cleaned_response = re.sub(r'\n?```$', '', cleaned_response)
        next_tasks = json.loads(cleaned_response)
        if not isinstance(next_tasks, list):
            next_tasks = []
    except json.JSONDecodeError:
        logger.error(f"Could not parse next tasks JSON: {response_text}")
        next_tasks = []
    
    return next_tasks

async def execute_plan(plan_obj: Dict[str, Any], email: str, origin_url: str, page_data: Dict[str, Any] = None, quiz_attempt: QuizAttempt = None) -> Dict[str, Any]:
    """Execute the LLM-generated plan with iterative task batches"""
    try:
        plan = plan_obj
        artifacts = {}
        execution_log = []
        all_tasks = plan.get("tasks", [])
        max_iterations = 5
        iteration = 0
        
        if quiz_attempt:
            quiz_attempt.plan = plan
        
        logger.info(f"Starting execution with iterative plan refinement (max {max_iterations} iterations)")
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"=== Iteration {iteration} ===")
            logger.info(f"Executing {len(all_tasks)} tasks")
            
            for task in all_tasks:
                task_id = task["id"]
                tool_name = task["tool_name"]
                inputs = task.get("inputs", {})
                produces = task.get("produces", [])
                
                logger.info(f"Executing task {task_id}: {tool_name}")
                
                try:
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
                    elif tool_name == "parse_csv":
                        result = parse_csv(inputs["path"])
                    elif tool_name == "parse_excel":
                        result = parse_excel(inputs["path"])
                    elif tool_name == "parse_json_file":
                        result = parse_json_file(inputs["path"])
                    elif tool_name == "parse_html_tables":
                        result = parse_html_tables(inputs["path_or_html"])
                    elif tool_name == "parse_pdf_tables":
                        result = parse_pdf_tables(inputs["path"], inputs.get("pages", "all"))
                    elif tool_name == "dataframe_ops":
                        result = dataframe_ops(inputs["op"], inputs.get("params", {}))
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
                                        processed_value = processed_value.replace("your secret", SECRET)
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
                        df = inputs["dataframe"]
                        result = AnalysisTools.calculate_statistics(df, inputs["columns"], inputs["stats"])
                        result = {"statistics": result}
                    elif tool_name == "apply_ml_model":
                        df = inputs["dataframe"]
                        result = AnalysisTools.apply_ml_model(df, inputs["model_type"], **inputs.get("kwargs", {}))
                        result = {"model_result": result}
                    elif tool_name == "geospatial_analysis":
                        df = inputs.get("dataframe")
                        result = AnalysisTools.geospatial_analysis(df, inputs["analysis_type"], **inputs.get("kwargs", {}))
                        result = {"analysis_result": result}
                    elif tool_name == "create_chart":
                        df = inputs["dataframe"]
                        result = VisualizationTools.create_chart(df, inputs["chart_type"], inputs["x_col"], inputs["y_col"], inputs.get("title", ""), inputs.get("output_path"))
                        result = {"chart_path": result}
                    elif tool_name == "create_interactive_chart":
                        df = inputs["dataframe"]
                        result = VisualizationTools.create_interactive_chart(df, inputs["chart_type"], inputs["x_col"], inputs["y_col"], inputs.get("title", ""), inputs.get("output_path"))
                        result = {"chart_path": result}
                    elif tool_name == "generate_narrative":
                        df = inputs["dataframe"]
                        result = VisualizationTools.generate_narrative(df, inputs.get("summary_stats", {}))
                        result = {"narrative": result}
                    elif tool_name == "create_presentation_slide":
                        result = VisualizationTools.create_presentation_slide(inputs["title"], inputs["content"], inputs.get("output_path"))
                        result = {"slide_path": result}
                    else:
                        raise ValueError(f"Unknown tool: {tool_name}")
                    
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
            
            logger.info("Checking if execution is complete...")
            
            if len(all_tasks) == 0:
                logger.info("Plan has 0 tasks - considered complete")
                break
            
            if page_data:
                completion_status = await check_plan_completion(plan, artifacts, page_data)
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
        
        logger.info("Preparing final answer for submission")
        
        if final_spec.get("from"):
            from_key = final_spec["from"]
            cleaned_key = from_key.strip()
            cleaned_key = re.sub(r"^static value\s*['\"]?|['\"]?$", "", cleaned_key)
            cleaned_key = re.sub(r"^['\"]|['\"]$", "", cleaned_key).strip()
            
            if from_key in artifacts:
                final_answer = artifacts[from_key]
                logger.info(f"Final answer extracted from artifact '{from_key}'")
            elif cleaned_key in artifacts:
                final_answer = artifacts[cleaned_key]
                logger.info(f"Final answer extracted from artifact '{cleaned_key}'")
            else:
                logger.info(f"Artifact '{from_key}' not found, using as literal answer: {cleaned_key}")
                final_answer = cleaned_key
        
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
        
        # Execute final submission
        submit_url = plan.get("submit_url")
        request_body_spec = plan.get("request_body", {})
        submission_result = None
        
        logger.info(f"[SUBMISSION_PREP] Submit URL: {submit_url}")
        logger.info(f"[SUBMISSION_PREP] Email: {email}")
        logger.info(f"[SUBMISSION_PREP] Origin URL: {origin_url}")
        
        if submit_url and request_body_spec:
            submission_body = {}
            email_key = request_body_spec.get("email_key", "email")
            submission_body[email_key] = email
            logger.info(f"[SUBMISSION_BUILD] Email key: {email_key}")
            
            secret_key = request_body_spec.get("secret_key", "secret")
            submission_body[secret_key] = SECRET
            logger.info(f"[SUBMISSION_BUILD] Secret key: {secret_key}")
            
            url_key = request_body_spec.get("url_value", "url")
            if url_key:
                submission_body[url_key] = origin_url
                logger.info(f"[SUBMISSION_BUILD] URL key: {url_key}")
            
            answer_key = request_body_spec.get("answer_key", "answer")
            submission_body[answer_key] = final_answer
            logger.info(f"[SUBMISSION_BUILD] Answer key: {answer_key}")
            logger.info(f"[SUBMISSION_BUILD] Final answer: {final_answer}")
            
            logger.info(f"[FINAL_SUBMISSION_BODY] {json.dumps(submission_body, indent=2)}")
            submission_result = await answer_submit(submit_url, submission_body)
            
            if quiz_attempt:
                quiz_attempt.submission_response = submission_result.get("response", {}) if submission_result else None
                quiz_attempt.correct = submission_result.get("response", {}).get("correct") if submission_result else None
                quiz_attempt.answer = final_answer
                quiz_attempt.artifacts = artifacts
                quiz_attempt.execution_log = execution_log
        
        return {
            "success": True,
            "final_answer": final_answer,
            "final_answer_type": final_spec.get("type"),
            "submission_result": submission_result,
            "execution_log": execution_log,
            "artifacts_count": len(artifacts)
        }
        
    except Exception as e:
        logger.error(f"Plan execution failed: {e}")
        if quiz_attempt:
            quiz_attempt.error = str(e)
        return {
            "success": False,
            "error": str(e),
            "execution_log": execution_log
        }
