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
from dotenv import load_dotenv
from playwright.async_api import async_playwright
import httpx
import pandas as pd
import matplotlib.pyplot as plt
from tools import ToolRegistry, ScrapingTools, CleansingTools, ProcessingTools, AnalysisTools, VisualizationTools
from models import QuizAttempt, QuizRun

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Global registry for dataframes
dataframe_registry = {}

def get_secret():
    """Get SECRET from environment, loaded via dotenv"""
    return os.getenv("SECRET")


def extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code blocks"""
    # Remove markdown code block markers
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    return text


async def render_page(url: str) -> Dict[str, Any]:
    """Render page with Playwright and extract content"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        
        # Wait for any dynamic content to render (up to 3 seconds)
        try:
            await page.wait_for_load_state("networkidle", timeout=3000)
        except:
            pass
        
        content = await page.content()
        text = await page.evaluate("() => document.body.innerText")
        links = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
        pres = await page.eval_on_selector_all("pre,code", "els => els.map(e => e.innerText)")
        
        # Try to extract any dynamically rendered content from divs with IDs
        rendered_divs = await page.eval_on_selector_all("div[id]", "els => els.map(e => ({id: e.id, text: e.innerText, html: e.innerHTML}))")
        
        await browser.close()
        return {
            "html": content,
            "text": text,
            "links": links,
            "code_blocks": pres,
            "rendered_divs": rendered_divs
        }

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

def parse_csv(path: str = None, url: str = None) -> Dict[str, Any]:
    """Parse CSV file from path or URL"""
    try:
        # If URL is provided, use it directly
        source = url if url else path
        if not source:
            raise ValueError("Either 'path' or 'url' must be provided")
        
        df = pd.read_csv(source)
        dataframe_registry[f"df_{len(dataframe_registry)}"] = df
        return {
            "dataframe_key": f"df_{len(dataframe_registry)-1}",
            "shape": df.shape,
            "columns": list(df.columns),
            "sample": df.head().to_dict()
        }
    except Exception as e:
        logger.error(f"Error parsing CSV from {source}: {e}")
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

async def call_llm_for_plan(page_data: Dict[str, Any], previous_attempts: List["QuizAttempt"] = None) -> str:
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
                reason = prev.submission_response.get('reason', 'N/A')
                previous_context += f"  - Reason for failure: {reason}\n"
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

IMPORTANT RULES:
1. Submit URL must be extracted from the page text
2. Tasks array contains the work to do (can be empty if answer is direct)
3. final_answer_spec.from references an artifact key from tasks OR a literal value
4. request_body keys are the NAMES of fields in the submission JSON
5. Return ONLY valid JSON, no markdown wrapper, no extra text

AVAILABLE TOOLS:
- render_js_page(url): Render page and get text/links/code
- fetch_text(url): Fetch text content
- download_file(url): Download binary file
- parse_csv(path OR url): Parse CSV from local path or URL
- parse_excel/parse_json_file/parse_html_tables/parse_pdf_tables: Parse files
- dataframe_ops(op, params): DataFrame operations
- make_plot(spec): Create chart
- zip_base64(paths): Create zip archive
- call_llm(prompt, system_prompt, max_tokens, temperature): Call LLM
"""
    
    prompt = f"""
QUIZ PAGE DATA:
Text: {page_data['text']}
Code blocks: {page_data['code_blocks']}
Links: {page_data['links']}
HTML preview: {page_data['html'][:500]}...{previous_context}

ANALYZE THIS QUIZ AND GENERATE THE EXECUTION PLAN JSON.

Requirements:
1. Find the submit URL (look for POST endpoint, /submit, /answer paths)
2. Identify what answer is required (read the instruction text)
3. Determine the answer type (boolean, number, string, json)
4. Plan minimal tasks needed (often zero if answer is direct)
5. Build the request_body structure with proper field names

OUTPUT ONLY THE JSON PLAN, NO OTHER TEXT.
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
        logger.info(f"[LLM_PLAN] Raw plan response: {plan_text[:500]}")
        return plan_text

async def check_plan_completion(plan_obj: Dict[str, Any], artifacts: Dict[str, Any], page_data: Dict[str, Any]) -> Dict[str, Any]:
    """Check if plan execution is complete"""
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
    available_tools = """
    AVAILABLE TOOLS (use ONLY these tool names):
    - render_js_page(url): Render page with JavaScript
    - fetch_text(url): Fetch text content from URL
    - download_file(url): Download binary file
    - parse_csv(path OR url): Parse CSV file from local path or URL
    - parse_excel(path): Parse Excel file
    - parse_json_file(path): Parse JSON file
    - parse_html_tables(html_content): Parse HTML tables
    - parse_pdf_tables(path, pages): Parse PDF tables
    - dataframe_ops(op, params): DataFrame operations
    - make_plot(spec): Create plot
    - zip_base64(paths): Create zip archive
    - call_llm(prompt, system_prompt, max_tokens, temperature): Call LLM
    """
    
    system_prompt = f"""You are an execution planner generating the NEXT BATCH of tasks needed to proceed.
    
    {available_tools}
    
    Return a JSON array of task objects only. Each task should have:
    {{ "id": "string", "tool_name": "string", "inputs": {{}}, "produces": [{{"key": "string", "type": "string"}}], "notes": "string" }}
    
    IMPORTANT: Only use tool names from the AVAILABLE TOOLS list above.
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
        
        Generate the NEXT BATCH of tasks to gather the missing information using ONLY available tools.
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
                        result = parse_csv(
                            path=inputs.get("path"),
                            url=inputs.get("url")
                        )
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
        
        # First try to get the specified artifact
        from_key = final_spec.get("from", "")
        candidate_artifacts = {}
        
        if from_key:
            cleaned_key = from_key.strip()
            cleaned_key = re.sub(r"^static value\s*['\"]?|['\"]?$", "", cleaned_key)
            cleaned_key = re.sub(r"^['\"]|['\"]$", "", cleaned_key).strip()
            
            if from_key in artifacts:
                candidate_artifacts[from_key] = artifacts[from_key]
            elif cleaned_key in artifacts:
                candidate_artifacts[cleaned_key] = artifacts[cleaned_key]
        
        # If the specified artifact is not useful (HTML/script/dict), look for better alternatives
        # Prioritize: extracted_*, rendered_*, scraped_*, secret_* artifacts that are meaningful
        if not candidate_artifacts or all(isinstance(v, dict) or (isinstance(v, str) and ('<' in v or 'script' in v.lower())) for v in candidate_artifacts.values()):
            logger.info("[ARTIFACT_SELECTION] Specified artifact not useful, searching for better alternatives...")
            
            # Sort to check newest artifacts first (rendered_*/scraped_* typically created last)
            for key in sorted(artifacts.keys(), reverse=True):
                value = artifacts[key]
                # Skip HTML, script tags, and dict objects
                if isinstance(value, dict):
                    continue
                if isinstance(value, str) and ('<' in value or 'script' in value.lower()):
                    continue
                # Prefer ANY meaningful artifact (not just extracted/secret)
                # This includes rendered_*, scraped_*, extracted_*, secret_*
                if isinstance(value, str) and len(value) < 500 and len(value) > 2:
                    prefixes = ('extracted_', 'rendered_', 'scraped_', 'secret_', 'result_', 'answer_')
                    if any(key.startswith(p) for p in prefixes):
                        logger.info(f"[ARTIFACT_SELECTION] Found alternative: {key} = {value[:100]}")
                        candidate_artifacts[key] = value
                        break  # Take the first (newest) good alternative
        
        # Select best artifact
        if candidate_artifacts:
            # Prefer most recently created/processed artifact (last in dict)
            final_answer = list(candidate_artifacts.values())[-1]
            selected_key = list(candidate_artifacts.keys())[-1]
            logger.info(f"Final answer extracted from artifact '{selected_key}'")
        elif from_key:
            logger.info(f"Artifact '{from_key}' not found, using as literal answer: {from_key}")
            final_answer = cleaned_key
        
        # Handle dict objects and string representations of dicts
        if isinstance(final_answer, dict):
            # Actual dict object
            if 'text' in final_answer:
                logger.info(f"[ARTIFACT_EXTRACTION] Extracting 'text' from dict object")
                final_answer = final_answer['text']
                logger.info(f"[ARTIFACT_EXTRACTION] Extracted: {str(final_answer)[:100]}...")
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
            submission_body[secret_key] = get_secret()
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


async def run_pipeline(email: str, url: str) -> Dict[str, Any]:
    """
    Run the complete quiz-solving pipeline for a single URL
    Handles multiple sequential quizzes and retry logic
    """
    try:
        logger.info(f"[PIPELINE] Starting pipeline for {email} at {url}")
        
        current_url = url
        quiz_chain = []
        quiz_runs = {}
        max_chain_iterations = 10
        
        while len(quiz_chain) < max_chain_iterations:
            logger.info(f"=== Processing Quiz: {current_url} ===")
            
            # Get or create quiz run tracker for this URL
            if current_url not in quiz_runs:
                quiz_runs[current_url] = QuizRun(current_url)
                logger.info(f"[QUIZ_RUN] New quiz run created for {current_url}")
            
            quiz_run = quiz_runs[current_url]
            
            # Start a new attempt
            quiz_attempt = quiz_run.start_attempt()
            logger.info(f"[QUIZ_RUN] Starting attempt {quiz_attempt.attempt_number}")
            
            # Render page
            logger.info(f"[QUIZ_RUN] Rendering page for attempt {quiz_attempt.attempt_number}")
            page_data = await render_page(current_url)
            logger.info(f"[QUIZ_RUN_PAGE] Rendered page - text length: {len(page_data.get('text', ''))}, links: {len(page_data.get('links', []))}")
            
            # Generate execution plan
            logger.info(f"[QUIZ_RUN] Generating execution plan for attempt {quiz_attempt.attempt_number}")
            plan_json_raw = await call_llm_for_plan(page_data, quiz_run.attempts[:-1])  # Pass previous attempts
            logger.info(f"[QUIZ_PLAN_RAW] Raw LLM response: {plan_json_raw[:300]}...")
            
            # Extract JSON from markdown wrapper
            plan_json = extract_json_from_markdown(plan_json_raw)
            
            # Parse and validate plan
            try:
                plan_obj = json.loads(plan_json)
                
                # Handle both old format (tasks) and new format (steps)
                # Convert steps to tasks for compatibility
                if "steps" in plan_obj and "tasks" not in plan_obj:
                    logger.info(f"[QUIZ_PLAN_CONVERT] Converting 'steps' format to 'tasks' format")
                    plan_obj["tasks"] = []
                
                tasks_count = len(plan_obj.get("tasks", []))
                logger.info(f"[QUIZ_RUN] Plan parsed successfully with {tasks_count} tasks")
                logger.info(f"[QUIZ_PLAN] Full plan: {json.dumps(plan_obj, indent=2)[:1000]}")
            except json.JSONDecodeError as e:
                logger.error(f"[QUIZ_RUN] Invalid JSON plan: {e}")
                logger.error(f"[QUIZ_PLAN_INVALID] Extracted JSON: {plan_json[:500]}")
                quiz_attempt.error = f"Invalid plan JSON: {e}"
                quiz_attempt.finish()
                quiz_run.finish_current_attempt()
                return {
                    "success": False,
                    "error": f"Invalid plan JSON: {e}",
                    "quiz_runs": {url: qr.to_dict() for url, qr in quiz_runs.items()}
                }
            
            # Store plan in attempt
            quiz_attempt.plan = plan_obj
            
            # If this is a retry, provide context from previous attempts
            previous_attempts_context = ""
            if quiz_attempt.attempt_number > 1:
                logger.info(f"[QUIZ_RUN] This is attempt {quiz_attempt.attempt_number}, analyzing previous attempts...")
                previous_attempts_context = "\n\nPrevious attempts:\n"
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
                    current_url = next_url
                    
                    # Add successful run to chain
                    quiz_chain.append({
                        "quiz_url": current_url,
                        "quiz_run": quiz_run.to_dict()
                    })
                else:
                    logger.info("[QUIZ_COMPLETE] No next quiz URL. Quiz chain complete!")
                    quiz_chain.append({
                        "quiz_url": current_url,
                        "quiz_run": quiz_run.to_dict()
                    })
                    break
            else:
                # Answer was incorrect, check if we can retry
                elapsed_time = quiz_run.elapsed_time_since_first()
                max_retry_time = 120  # 2 minutes in seconds
                max_attempts = 3  # Maximum 3 attempts per quiz
                
                logger.info(f"[QUIZ_RETRY] Answer was incorrect. Elapsed time: {elapsed_time:.1f}s / Max: {max_retry_time}s, Attempts: {quiz_attempt.attempt_number}/{max_attempts}")
                
                # Check both time limit and attempt limit
                if quiz_attempt.attempt_number >= max_attempts:
                    logger.error(f"[QUIZ_FAILED] Max attempts ({max_attempts}) reached. Quiz failed after {quiz_attempt.attempt_number} attempts.")
                    quiz_chain.append({
                        "quiz_url": current_url,
                        "quiz_run": quiz_run.to_dict(),
                        "failed": True,
                        "reason": f"Max attempts ({max_attempts}) exceeded"
                    })
                    break
                elif quiz_run.can_retry(max_retry_time):
                    logger.info(f"[QUIZ_RETRY] Can retry! Attempting again...")
                    # Loop continues to next attempt
                    continue
                else:
                    logger.error(f"[QUIZ_FAILED] Max retry time exceeded. Quiz failed after {quiz_attempt.attempt_number} attempts.")
                    quiz_chain.append({
                        "quiz_url": current_url,
                        "quiz_run": quiz_run.to_dict(),
                        "failed": True,
                        "reason": "Max retry time exceeded"
                    })
                    break
        
        logger.info(f"[PIPELINE_COMPLETE] Quiz chain complete. Solved {len(quiz_chain)} quizzes")
        
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
