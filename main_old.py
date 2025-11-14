# Windows compatibility for asyncio
import asyncio
if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# main.py
import os, time, json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, ValidationError, HttpUrl
from dotenv import load_dotenv
from typing import Any, Dict
from playwright.async_api import async_playwright
import httpx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tempfile import NamedTemporaryFile, TemporaryDirectory
import base64
import io
import zipfile
from typing import Dict, Any, List, Optional
import logging
import lxml
import html5lib
from tools import ToolRegistry, ScrapingTools, CleansingTools, ProcessingTools, AnalysisTools, VisualizationTools

load_dotenv()  # Load .env if present

app = FastAPI(title="Quiz Solver API")

SECRET = os.getenv("SECRET")

if not SECRET:
    # Fail fast if secret not configured
    raise RuntimeError("Environment variable SECRET is not set.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global registry for dataframes
dataframe_registry = {}

# ===== QUIZ ATTEMPT TRACKING =====

class QuizAttempt:
    """Tracks a single attempt at solving a quiz"""
    def __init__(self, attempt_number: int):
        self.attempt_number = attempt_number
        self.start_time = time.time()
        self.end_time = None
        self.plan = None
        self.answer = None
        self.submission_response = None
        self.correct = None
        self.error = None
        self.artifacts = {}
        self.execution_log = []
        
    def finish(self):
        self.end_time = time.time()
    
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "plan_summary": {
                "tasks_count": len(self.plan.get("tasks", [])) if self.plan else 0,
                "final_answer_spec": self.plan.get("final_answer_spec") if self.plan else None
            },
            "answer": str(self.answer)[:200] if self.answer else None,
            "correct": self.correct,
            "error": self.error,
            "artifacts_count": len(self.artifacts),
            "execution_log_lines": len(self.execution_log)
        }

class QuizRun:
    """Tracks all attempts for a single quiz URL"""
    def __init__(self, quiz_url: str):
        self.quiz_url = quiz_url
        self.first_attempt_time = None
        self.attempts = []
        self.current_attempt = None
        
    def start_attempt(self) -> QuizAttempt:
        if not self.first_attempt_time:
            self.first_attempt_time = time.time()
        self.current_attempt = QuizAttempt(len(self.attempts) + 1)
        self.attempts.append(self.current_attempt)
        logger.info(f"[QUIZ_RUN] Starting attempt {self.current_attempt.attempt_number} for quiz {self.quiz_url[:50]}...")
        return self.current_attempt
    
    def finish_current_attempt(self):
        if self.current_attempt:
            self.current_attempt.finish()
    
    def elapsed_time_since_first(self) -> float:
        if not self.first_attempt_time:
            return 0
        return time.time() - self.first_attempt_time
    
    def can_retry(self, max_time_seconds: float = 150) -> bool:
        """Check if we can retry (time limit: 150 seconds = 2.5 minutes)"""
        return self.elapsed_time_since_first() < max_time_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quiz_url": self.quiz_url,
            "first_attempt_time": self.first_attempt_time,
            "total_elapsed": self.elapsed_time_since_first(),
            "attempts_count": len(self.attempts),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "success": any(a.correct for a in self.attempts) if self.attempts else False
        }

# ===== TOOL IMPLEMENTATIONS =====

async def render_page(url: str) -> Dict[str, Any]:
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
    """Tool 2: Fetch text content from URL"""
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
    """Tool 3: Download file and return metadata"""
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
    """Tool 4a: Parse CSV file"""
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
    """Tool 4b: Parse Excel file"""
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
    """Tool 4c: Parse JSON file"""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return {"data": data, "type": type(data).__name__}
    except Exception as e:
        logger.error(f"Error parsing JSON {path}: {e}")
        raise

def parse_html_tables(html_content: str) -> Dict[str, Any]:
    """Tool 4d: Parse HTML tables"""
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
    """Tool 4e: Parse PDF tables (placeholder - requires pdfplumber)"""
    try:
        # Note: This requires pdfplumber installation
        # For now, return placeholder
        return {
            "warning": "PDF parsing requires pdfplumber installation",
            "path": path,
            "pages": pages
        }
    except Exception as e:
        logger.error(f"Error parsing PDF {path}: {e}")
        raise

def dataframe_ops(op: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Tool 5: Perform DataFrame operations"""
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
            # Simple filtering - in production, use safer eval or predefined conditions
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
        
        # Store result if it's a DataFrame
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
    """Tool 6: Create plot and return base64 data URI"""
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
            # Simple plot from data
            data = spec.get("data", [])
            plt.plot(data)
        
        plt.title(spec.get("title", "Plot"))
        plt.xlabel(spec.get("xlabel", ""))
        plt.ylabel(spec.get("ylabel", ""))
        
        if spec.get("grid"):
            plt.grid(True)
        
        # Convert to base64
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
    """Tool 7: Zip files and return base64 data URI"""
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zip_file:
            for path in paths:
                zip_file.write(path, os.path.basename(path))
        
        buf.seek(0)
        zip_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return {"base64_uri": f"data:application/zip;base64,{zip_base64}"}
        
    except Exception as e:
        logger.error(f"Error creating zip: {e}")
        raise

async def answer_submit(url: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Tool 0: Submit answer to quiz endpoint"""
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
    """Generic LLM call function
    
    Args:
        prompt: The user prompt to send to the LLM
        system_prompt: Optional system prompt. If not provided, uses a generic one.
        max_tokens: Maximum tokens in the response (default: 2000)
        temperature: Temperature for sampling (default: 0, deterministic)
    
    Returns:
        The LLM's response text
    """
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
    logger.info(f"[LLM_CALL] Max tokens: {max_tokens}, Temperature: {temperature}")
    
    async with httpx.AsyncClient() as client:
        logger.info(f"[LLM_REQUEST] POST {OPEN_AI_BASE_URL}")
        response = await client.post(OPEN_AI_BASE_URL, headers=headers, json=json_data, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer_text = data["choices"][0]["message"]["content"]
        
        logger.info(f"[LLM_RESPONSE] Status: 200")
        logger.info(f"[LLM_RESPONSE] Content: {answer_text[:300]}...")
        logger.info(f"[LLM_RESPONSE] Length: {len(answer_text)} chars")
        
        return answer_text.strip()

async def call_llm_for_plan(page_data: Dict[str, Any]) -> str:
    system_prompt = """
        You are an execution planner for an automated quiz-solving agent. Your job is to read a rendered quiz page (HTML, visible text, links, code/pre blocks), infer the required steps, and emit a machine-executable plan as strict JSON. Do not perform the tasks yourself. Only plan and specify tools and inputs.

        Constraints:

        Use only URLs present in the provided page content (including those revealed by JS like atob replacements). Do not invent URLs.
        Respect time budget: prefer minimal, deterministic steps.
        The final answer type must match the quiz instruction: boolean, number, string, base64_uri, or json.
        If instructions are ambiguous, state the assumption in the plan’s notes.
        Output must be valid JSON matching the schema provided. No extra prose.
        You have access to these tools (names are identifiers used in the plan):

        1 render_js_page(url): Render JS, return {html, text, links, code_blocks}.
        2 fetch_text(url): GET, return text and headers.
        3 download_file(url): GET binary, return {path, content_type, size}.
        4 parse_csv(path), parse_excel(path), parse_json_file(path), parse_html_tables(path_or_html), parse_pdf_tables(path, pages).
        5 dataframe_ops(op, params): Perform operations like select/filter/sum/groupby on DataFrames registry keys.
        6 make_plot(spec): Return data URI base64 image.
        7 zip_base64(paths): Return data URI base64 zip.
        8 call_llm(prompt, system_prompt, max_tokens, temperature): Call LLM with given prompt and return text.

        if you feel like you need other tools, explain in notes, they will be added later.

        Output JSON Schema
        { "submit_url": "string", "origin_url": "string", "tasks": [ { "id": "string", "tool_name": "string", "inputs": {}, "produces": [{"key": "string", "type": "string"}], "notes": "string" } ], "final_answer_spec": { "type": "boolean|number|string|base64_uri|json", "from": "key or expression referencing produced artifacts", "json_schema": {} }, "request_body": { "email_key": "string", "secret_key": "string", "url_value": "string", "answer_key": "string (optional, key name for the answer in request body)" }, "assumptions": "string" }

        Rules:

        tasks[i].produces[].key are variable names you will reference later.
        If no data download/parsing is required, minimize tasks.
        request_body.url_value must be the KEY NAME for the quiz page URL in the submission payload (e.g., "url" or "target_url"). The actual URL value will be filled in at execution time.
        submit_url must be exactly as found on the page.
        """
    prompt = f"""
        You are given the rendered quiz page extraction:

        html: {page_data['html']}

        text: {page_data['text']}

        code_blocks: {page_data['code_blocks']}

        links: {page_data['links']}
  
        Your job:

        Identify the submit URL.
        Identify the URL value that must be echoed in the request body (if present).
        Identify the instruction for the answer and the required answer type.
        Plan the minimal tasks to produce the answer using only allowed tools and provided URLs.
        Produce the final JSON .

        Important patterns to recognize:

        Some pages replace placeholders like window.location.origin; use the rendered values in the provided text/html.
        Instructions may be inside code/pre blocks or base64-decoded JS. Use the visible text/code after rendering.
        If instruction is about a dataset (CSV/Excel/HTML/PDF), plan to download and parse accordingly, then compute the requested metric.
        Ensure payload will be under 1MB; keep attachments small.
        Remember: Output only the JSON plan per schema.

        Notes:

        If multiple candidates for submit URL are present, pick the one explicitly stated by “POST this JSON to ...”.
        Guardrails
        If submit URL or url_value cannot be confidently found, set assumptions to explain and leave them blank; the executor will fail fast with a clear error.
        Never invent or alter domains; copy exactly from the page’s visible text.
        Keep the plan minimal to meet the 3-minute SLA.
        """
    
    OPEN_AI_BASE_URL = "https://aipipe.org/openai/v1/chat/completions"
    API_KEY = os.getenv("API_KEY")
    import httpx
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

async def call_llm_for_answer(plan_obj: Dict[str, Any], artifacts: Dict[str, Any], page_data: Dict[str, Any]) -> str:
    """Call LLM to generate the final answer based on task artifacts"""
    system_prompt = """
        You are an answer generator for an automated quiz-solving agent. Your job is to read:
        1. The original quiz page data (HTML, visible text, instructions)
        2. The task execution plan that was executed
        3. The artifacts produced by each task
        
        Then, generate the final answer that satisfies the quiz requirements.
        
        Constraints:
        - The answer type must match the quiz instruction: boolean, number, string, base64_uri, or json
        - Only use the artifacts provided; do not invent data
        - If instructions are ambiguous, state your assumption
        - Output only the answer value, unless JSON/complex format is required per instructions
        
        Artifact Reference:
        You have access to all artifacts produced by the executed tasks. Each artifact key maps to specific data:
        - For parsed files: keys like "data", "dataframe_key", "shape", "columns", "sample"
        - For operations: keys like "result", "base64_uri"
        - Use these to derive the final answer
        """
    
    # Serialize artifacts carefully, excluding non-JSON-serializable types
    artifacts_summary = {}
    for key, value in artifacts.items():
        if isinstance(value, (str, int, float, bool, list, dict)):
            artifacts_summary[key] = value
        else:
            artifacts_summary[key] = str(value)
    
    prompt = f"""
        Quiz Page Data:
        - Text: {page_data.get('text', 'N/A')}
        - Code Blocks: {page_data.get('code_blocks', 'N/A')}
        
        Executed Plan:
        {json.dumps(plan_obj, indent=2)}
        
        Produced Artifacts:
        {json.dumps(artifacts_summary, indent=2)}
        
        Please analyze the quiz instructions, the plan that was executed, and the artifacts produced.
        Generate the final answer that matches the answer type specified in final_answer_spec.type.
        
        If the answer type is:
        - boolean: output "true" or "false"
        - number: output the numeric value
        - string: output the string value
        - base64_uri: output the data URI
        - json: output a valid JSON object/array
        
        Return ONLY the answer value with no additional text or explanation.
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
        "max_tokens": 2000,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(OPEN_AI_BASE_URL, headers=headers, json=json_data, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer_text = data["choices"][0]["message"]["content"]
        return answer_text.strip()

async def check_plan_completion(plan_obj: Dict[str, Any], artifacts: Dict[str, Any], page_data: Dict[str, Any]) -> Dict[str, Any]:
    """Check if the current plan execution is complete or if more tasks are needed.
    
    Returns:
        {
            "needs_more_tasks": bool,
            "answer_ready": bool,
            "next_tasks": [] (if needs_more_tasks is True),
            "reason": "explanation of why more tasks are needed or why we're done"
        }
    """
    system_prompt = """
        You are a task planner evaluating execution progress. Given the current artifacts and plan execution,
        determine if:
        1. The answer is ready to submit (final answer can be extracted from artifacts)
        2. More tasks need to be executed to get the answer
        
        Respond with a JSON object:
        {
            "answer_ready": boolean,
            "needs_more_tasks": boolean,
            "reason": "explanation",
            "recommended_next_action": "what should be done next"
        }
        
        If answer_ready is true, set needs_more_tasks to false.
        If more tasks are needed, explain what data is missing and what tools would help.
        """
    
    # Serialize artifacts
    artifacts_summary = {}
    for key, value in artifacts.items():
        if isinstance(value, (str, int, float, bool, list, dict)):
            artifacts_summary[key] = value
        else:
            artifacts_summary[key] = str(value)[:500]  # Truncate long values
    
    prompt = f"""
        Current Plan:
        {json.dumps(plan_obj.get('final_answer_spec', {}), indent=2)}
        
        Produced Artifacts:
        {json.dumps(artifacts_summary, indent=2)}
        
        Quiz Instructions (from page text):
        {page_data.get('text', 'N/A')[:1000]}
        
        Question: Based on the artifacts we have so far, can we extract the final answer?
        If not, what additional data or tasks do we need?
        """
    
    response_text = await call_llm(prompt, system_prompt, 1000, 0)
    
    try:
        # Strip markdown code blocks if present
        import re
        cleaned_response = response_text.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response)  # Remove opening ```json
        cleaned_response = re.sub(r'\n?```$', '', cleaned_response)  # Remove closing ```
        
        # Try to parse JSON response
        result = json.loads(cleaned_response)
    except json.JSONDecodeError:
        # If not JSON, assume we need more tasks
        result = {
            "answer_ready": False,
            "needs_more_tasks": True,
            "reason": response_text,
            "recommended_next_action": response_text
        }
    
    return result

async def generate_next_tasks(plan_obj: Dict[str, Any], artifacts: Dict[str, Any], page_data: Dict[str, Any], completion_status: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate the next batch of tasks based on completion analysis.
    
    Returns:
        List of new task objects to append to the plan
    """
    system_prompt = """
        You are an execution planner for an automated quiz-solving agent. Based on the current execution status,
        generate the NEXT BATCH of tasks needed to proceed.
        
        Available tools:
        0 answer_submit(url, body): POST JSON to submit URL.
        1 render_js_page(url): Render JS, return {html, text, links, code_blocks}.
        2 fetch_text(url): GET, return text and headers.
        3 download_file(url): GET binary, return {path, content_type, size}.
        4 parse_csv(path), parse_excel(path), parse_json_file(path), parse_html_tables(path_or_html), parse_pdf_tables(path, pages).
        5 dataframe_ops(op, params): Perform operations on DataFrames.
        6 make_plot(spec): Return data URI base64 image.
        7 zip_base64(paths): Return data URI base64 zip.
        8 call_llm(prompt, system_prompt, max_tokens, temperature): Call LLM with prompt.
        
        Return a JSON array of task objects. Each task should have:
        { "id": "string", "tool_name": "string", "inputs": {}, "produces": [{"key": "string", "type": "string"}], "notes": "string" }
        """
    
    # Serialize current state
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
        Code Blocks: {page_data.get('code_blocks', [])[:3]}
        
        Generate the NEXT BATCH of tasks (as a JSON array) to gather the missing information.
        Focus on what's needed based on the recommended_next_action.
        Return ONLY valid JSON array, no other text.
        """
    
    response_text = await call_llm(prompt, system_prompt, 2000, 0)
    
    try:
        # Strip markdown code blocks if present
        import re
        cleaned_response = response_text.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response)  # Remove opening ```json
        cleaned_response = re.sub(r'\n?```$', '', cleaned_response)  # Remove closing ```
        
        next_tasks = json.loads(cleaned_response)
        if not isinstance(next_tasks, list):
            next_tasks = []
    except json.JSONDecodeError:
        logger.error(f"Could not parse next tasks JSON: {response_text}")
        next_tasks = []
    
    return next_tasks
    
# ===== PLAN EXECUTION ENGINE =====

async def execute_plan(plan_obj: Dict[str, Any], email: str, origin_url: str, page_data: Dict[str, Any] = None, quiz_attempt: QuizAttempt = None) -> Dict[str, Any]:
    """Execute the LLM-generated plan with iterative task batches"""
    try:
        plan = plan_obj
        artifacts = {}
        execution_log = []
        all_tasks = plan.get("tasks", [])
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        # Store plan in attempt tracker if available
        if quiz_attempt:
            quiz_attempt.plan = plan
        
        logger.info(f"Starting execution with iterative plan refinement (max {max_iterations} iterations)")
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"=== Iteration {iteration} ===")
            logger.info(f"Executing {len(all_tasks)} tasks")
            
            # Execute each task in this batch
            for task in all_tasks:
                task_id = task["id"]
                tool_name = task["tool_name"]
                inputs = task.get("inputs", {})
                produces = task.get("produces", [])
                notes = task.get("notes", "")
                
                logger.info(f"Executing task {task_id}: {tool_name}")
                
                try:
                    # Execute tool based on name (same as before)
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
                        logger.info(f"[TOOL_EXEC] {task_id}: parse_csv - Path: {inputs['path']}")
                        result = parse_csv(inputs["path"])
                        logger.info(f"[TOOL_RESULT] {task_id}: parse_csv - Shape: {result.get('shape')}")
                    elif tool_name == "parse_excel":
                        logger.info(f"[TOOL_EXEC] {task_id}: parse_excel - Path: {inputs['path']}")
                        result = parse_excel(inputs["path"])
                        logger.info(f"[TOOL_RESULT] {task_id}: parse_excel - Shape: {result.get('shape')}")
                    elif tool_name == "parse_json_file":
                        logger.info(f"[TOOL_EXEC] {task_id}: parse_json_file - Path: {inputs['path']}")
                        result = parse_json_file(inputs["path"])
                        logger.info(f"[TOOL_RESULT] {task_id}: parse_json_file - Type: {result.get('type')}")
                    elif tool_name == "parse_html_tables":
                        logger.info(f"[TOOL_EXEC] {task_id}: parse_html_tables")
                        result = parse_html_tables(inputs["path_or_html"])
                        logger.info(f"[TOOL_RESULT] {task_id}: parse_html_tables - Table count: {len(result.get('tables', {}))}")
                    elif tool_name == "parse_pdf_tables":
                        logger.info(f"[TOOL_EXEC] {task_id}: parse_pdf_tables - Path: {inputs['path']}")
                        result = parse_pdf_tables(inputs["path"], inputs.get("pages", "all"))
                        logger.info(f"[TOOL_RESULT] {task_id}: parse_pdf_tables - Result: {result}")
                    elif tool_name == "dataframe_ops":
                        logger.info(f"[TOOL_EXEC] {task_id}: dataframe_ops - Op: {inputs['op']}")
                        result = dataframe_ops(inputs["op"], inputs.get("params", {}))
                        logger.info(f"[TOOL_RESULT] {task_id}: dataframe_ops - Result type: {type(result).__name__}")
                    elif tool_name == "make_plot":
                        logger.info(f"[TOOL_EXEC] {task_id}: make_plot - Type: {inputs['spec'].get('type')}")
                        result = make_plot(inputs["spec"])
                        logger.info(f"[TOOL_RESULT] {task_id}: make_plot - Generated base64 URI")
                    elif tool_name == "zip_base64":
                        logger.info(f"[TOOL_EXEC] {task_id}: zip_base64 - File count: {len(inputs['paths'])}")
                        result = zip_base64(inputs["paths"])
                        logger.info(f"[TOOL_RESULT] {task_id}: zip_base64 - Generated base64 URI")
                    elif tool_name == "call_llm":
                        logger.info(f"[TOOL_EXEC] {task_id}: call_llm - Extraction task")
                        # Replace template variables in prompt with artifact values
                        prompt = inputs["prompt"]
                        import re
                        for artifact_key in artifacts:
                            artifact_value = str(artifacts[artifact_key])
                            # Replace both {{key}} and {key} patterns
                            prompt = re.sub(r'\{\{' + re.escape(artifact_key) + r'\}\}', artifact_value, prompt)
                            prompt = re.sub(r'\{' + re.escape(artifact_key) + r'\}', artifact_value, prompt)
                        logger.info(f"[TOOL_PROMPT] {task_id}: {prompt[:300]}...")
                        
                        # Add strict output instruction if not already present
                        system_prompt = inputs.get("system_prompt", "You are a helpful assistant.")
                        if "ONLY the" not in system_prompt and "only the" not in system_prompt:
                            system_prompt += "\n\nIMPORTANT: Return ONLY the extracted value, no explanations or additional text."
                        
                        result = await call_llm(prompt, system_prompt, inputs.get("max_tokens", 2000), inputs.get("temperature", 0))
                        logger.info(f"[TOOL_RESULT] {task_id}: call_llm - Response: {result[:200]}...")
                    elif tool_name == "answer_submit":
                        # For answer_submit during execution (not final submission)
                        submit_url = inputs["url"]
                        submit_body = inputs["body"]
                        
                        logger.info(f"[TOOL_EXEC] {task_id}: answer_submit - URL: {submit_url}")
                        
                        # Validate and correct the URL if needed
                        # If the submit body has a "url" field, it should be the current quiz page
                        if isinstance(submit_body, dict) and "url" in submit_body:
                            body_url = str(submit_body["url"]).strip()
                            # If body URL looks like a data endpoint (not a quiz page), replace with origin_url
                            # Check if it's a data endpoint OR doesn't match the current quiz page base URL
                            origin_base = origin_url.split('?')[0]  # Get base URL without query params
                            is_data_endpoint = "/data" in body_url or "/demo-scrape-data" in body_url
                            is_wrong_url = body_url and body_url != origin_url and not body_url.startswith(origin_base)
                            
                            if (is_data_endpoint or is_wrong_url or body_url == "this page's URL"):
                                logger.info(f"[TOOL_CORRECT] Replacing incorrect URL in body from {body_url} to {origin_url}")
                                submit_body["url"] = origin_url
                        
                        # Process template variables in the body
                        import re
                        if isinstance(submit_body, dict):
                            processed_body = {}
                            for key, value in submit_body.items():
                                if isinstance(value, str):
                                    processed_value = value
                                    
                                    # First: Replace artifact references {artifact_key} and {{artifact_key}}
                                    for artifact_key in artifacts:
                                        artifact_value = str(artifacts[artifact_key])
                                        # Replace both {{key}} and {key} patterns - EXACT MATCH ONLY
                                        processed_value = re.sub(r'\{\{' + re.escape(artifact_key) + r'\}\}', artifact_value, processed_value)
                                        processed_value = re.sub(r'\{' + re.escape(artifact_key) + r'\}', artifact_value, processed_value)
                                    
                                    # Second: Check if value is literally an artifact key name (LLM might use plain key names)
                                    # Only do this if the value is EXACTLY an artifact key
                                    if processed_value in artifacts:
                                        processed_value = str(artifacts[processed_value])
                                        logger.info(f"[TOOL_ARTIFACT_RESOLVE] {key}: Resolved literal artifact key '{processed_value}' to value")
                                    
                                    # Third: Replace special placeholder strings (exact match, word boundaries)
                                    # Only replace "your secret" as complete phrase
                                    if "your secret" in processed_value:
                                        processed_value = processed_value.replace("your secret", SECRET)
                                    
                                    # Only replace "this page's URL" as complete phrase
                                    if "this page's URL" in processed_value:
                                        processed_value = processed_value.replace("this page's URL", origin_url)
                                    
                                    # Only replace if it's literally just "email" (for email_key field values)
                                    # Don't replace "email" embedded in URLs or other values
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
                    
                    # New Modular Tools - Scraping
                    elif tool_name == "scrape_with_javascript":
                        logger.info(f"[TOOL_EXEC] {task_id}: scrape_with_javascript - URL: {inputs['url']}")
                        result = await ScrapingTools.scrape_with_javascript(
                            inputs["url"], 
                            inputs.get("wait_for_selector"),
                            inputs.get("timeout", 30000)
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: scrape_with_javascript - HTML length: {len(result)}")
                        result = {"content": result}
                    elif tool_name == "fetch_from_api":
                        logger.info(f"[TOOL_EXEC] {task_id}: fetch_from_api - URL: {inputs['url']}")
                        result = await ScrapingTools.fetch_from_api(
                            inputs["url"],
                            inputs.get("method", "GET"),
                            inputs.get("headers"),
                            inputs.get("body"),
                            inputs.get("timeout", 30)
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: fetch_from_api - Status: {result.get('status_code')}")
                    elif tool_name == "extract_html_text":
                        logger.info(f"[TOOL_EXEC] {task_id}: extract_html_text")
                        result = await ScrapingTools.extract_html_text(
                            inputs["html"],
                            inputs.get("selector")
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: extract_html_text - Text length: {len(result)}")
                        result = {"text": result}
                    
                    # New Modular Tools - Cleansing
                    elif tool_name == "clean_text":
                        logger.info(f"[TOOL_EXEC] {task_id}: clean_text")
                        result = CleansingTools.clean_text(
                            inputs["text"],
                            inputs.get("lowercase", False),
                            inputs.get("remove_special", True),
                            inputs.get("remove_whitespace", True)
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: clean_text - Length: {len(result)}")
                        result = {"cleaned_text": result}
                    elif tool_name == "extract_from_pdf":
                        logger.info(f"[TOOL_EXEC] {task_id}: extract_from_pdf - Path: {inputs['pdf_path']}")
                        result = CleansingTools.extract_from_pdf(
                            inputs["pdf_path"],
                            inputs.get("pages")
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: extract_from_pdf - Text length: {len(result)}")
                        result = {"text": result}
                    elif tool_name == "parse_csv_data":
                        logger.info(f"[TOOL_EXEC] {task_id}: parse_csv_data")
                        df = CleansingTools.parse_csv_data(
                            inputs["csv_content"],
                            inputs.get("delimiter", ",")
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: parse_csv_data - Shape: {df.shape}")
                        result = {"dataframe": df, "shape": str(df.shape)}
                    elif tool_name == "parse_json_data":
                        logger.info(f"[TOOL_EXEC] {task_id}: parse_json_data")
                        result = CleansingTools.parse_json_data(inputs["json_content"])
                        logger.info(f"[TOOL_RESULT] {task_id}: parse_json_data - Type: {type(result).__name__}")
                        result = {"data": result}
                    elif tool_name == "extract_structured_data":
                        logger.info(f"[TOOL_EXEC] {task_id}: extract_structured_data")
                        result = CleansingTools.extract_structured_data(
                            inputs["text"],
                            inputs["pattern"]
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: extract_structured_data - Matches: {len(result)}")
                        result = {"matches": result}
                    
                    # New Modular Tools - Processing
                    elif tool_name == "transform_dataframe":
                        logger.info(f"[TOOL_EXEC] {task_id}: transform_dataframe")
                        df = inputs["dataframe"]
                        result = ProcessingTools.transform_dataframe(
                            df,
                            inputs["operations"]
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: transform_dataframe - Shape: {result.shape}")
                        result = {"dataframe": result, "shape": str(result.shape)}
                    elif tool_name == "aggregate_data":
                        logger.info(f"[TOOL_EXEC] {task_id}: aggregate_data")
                        df = inputs["dataframe"]
                        result = ProcessingTools.aggregate_data(
                            df,
                            inputs["group_by"],
                            inputs["aggregations"]
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: aggregate_data - Shape: {result.shape}")
                        result = {"dataframe": result, "shape": str(result.shape)}
                    elif tool_name == "reshape_data":
                        logger.info(f"[TOOL_EXEC] {task_id}: reshape_data")
                        df = inputs["dataframe"]
                        result = ProcessingTools.reshape_data(
                            df,
                            inputs["reshape_type"],
                            **inputs.get("kwargs", {})
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: reshape_data - Shape: {result.shape}")
                        result = {"dataframe": result, "shape": str(result.shape)}
                    elif tool_name == "transcribe_content":
                        logger.info(f"[TOOL_EXEC] {task_id}: transcribe_content")
                        result = ProcessingTools.transcribe_content(
                            inputs["content"],
                            inputs.get("format_type", "text")
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: transcribe_content - Length: {len(result)}")
                        result = {"transcribed": result}
                    
                    # New Modular Tools - Analysis
                    elif tool_name == "filter_data":
                        logger.info(f"[TOOL_EXEC] {task_id}: filter_data")
                        df = inputs["dataframe"]
                        result = AnalysisTools.filter_data(
                            df,
                            inputs["filters"]
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: filter_data - Shape: {result.shape}")
                        result = {"dataframe": result, "shape": str(result.shape)}
                    elif tool_name == "sort_data":
                        logger.info(f"[TOOL_EXEC] {task_id}: sort_data")
                        df = inputs["dataframe"]
                        result = AnalysisTools.sort_data(
                            df,
                            inputs["sort_by"]
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: sort_data - Shape: {result.shape}")
                        result = {"dataframe": result, "shape": str(result.shape)}
                    elif tool_name == "calculate_statistics":
                        logger.info(f"[TOOL_EXEC] {task_id}: calculate_statistics")
                        df = inputs["dataframe"]
                        result = AnalysisTools.calculate_statistics(
                            df,
                            inputs["columns"],
                            inputs["stats"]
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: calculate_statistics - Stats calculated")
                        result = {"statistics": result}
                    elif tool_name == "apply_ml_model":
                        logger.info(f"[TOOL_EXEC] {task_id}: apply_ml_model - Model: {inputs['model_type']}")
                        df = inputs["dataframe"]
                        result = AnalysisTools.apply_ml_model(
                            df,
                            inputs["model_type"],
                            **inputs.get("kwargs", {})
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: apply_ml_model - Model applied")
                        result = {"model_result": result}
                    elif tool_name == "geospatial_analysis":
                        logger.info(f"[TOOL_EXEC] {task_id}: geospatial_analysis")
                        df = inputs.get("dataframe")
                        result = AnalysisTools.geospatial_analysis(
                            df,
                            inputs["analysis_type"],
                            **inputs.get("kwargs", {})
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: geospatial_analysis - Analysis complete")
                        result = {"analysis_result": result}
                    
                    # New Modular Tools - Visualization
                    elif tool_name == "create_chart":
                        logger.info(f"[TOOL_EXEC] {task_id}: create_chart - Type: {inputs['chart_type']}")
                        df = inputs["dataframe"]
                        result = VisualizationTools.create_chart(
                            df,
                            inputs["chart_type"],
                            inputs["x_col"],
                            inputs["y_col"],
                            inputs.get("title", ""),
                            inputs.get("output_path")
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: create_chart - Chart created")
                        result = {"chart_path": result}
                    elif tool_name == "create_interactive_chart":
                        logger.info(f"[TOOL_EXEC] {task_id}: create_interactive_chart - Type: {inputs['chart_type']}")
                        df = inputs["dataframe"]
                        result = VisualizationTools.create_interactive_chart(
                            df,
                            inputs["chart_type"],
                            inputs["x_col"],
                            inputs["y_col"],
                            inputs.get("title", ""),
                            inputs.get("output_path")
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: create_interactive_chart - Chart created")
                        result = {"chart_path": result}
                    elif tool_name == "generate_narrative":
                        logger.info(f"[TOOL_EXEC] {task_id}: generate_narrative")
                        df = inputs["dataframe"]
                        result = VisualizationTools.generate_narrative(
                            df,
                            inputs.get("summary_stats", {})
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: generate_narrative - Narrative generated")
                        result = {"narrative": result}
                    elif tool_name == "create_presentation_slide":
                        logger.info(f"[TOOL_EXEC] {task_id}: create_presentation_slide")
                        result = VisualizationTools.create_presentation_slide(
                            inputs["title"],
                            inputs["content"],
                            inputs.get("output_path")
                        )
                        logger.info(f"[TOOL_RESULT] {task_id}: create_presentation_slide - Slide created")
                        result = {"slide_path": result}
                    
                    else:
                        raise ValueError(f"Unknown tool: {tool_name}")
                    
                    # Store produced artifacts
                    for produce in produces:
                        key = produce["key"]
                        # Handle different result types
                        if isinstance(result, dict) and key in result:
                            # If result is a dict and has the key, use that value
                            artifacts[key] = result[key]
                        elif isinstance(result, (str, int, float, bool, list)):
                            # For string results from call_llm, try to extract just the value
                            if tool_name == "call_llm" and isinstance(result, str):
                                # Clean up LLM response - remove common explanation patterns
                                import re
                                cleaned = result.strip()
                                # Remove common prefixes like "The secret code is:" or "The answer is:"
                                cleaned = re.sub(r'^.*?(?:is|are|be|found|extracted|as):\s*', '', cleaned, flags=re.IGNORECASE)
                                # Remove markdown code blocks
                                cleaned = re.sub(r'^```.*?\n?', '', cleaned)
                                cleaned = re.sub(r'\n?```$', '', cleaned)
                                # Remove trailing explanations
                                cleaned = re.sub(r'\s*[.!?]\s+.*$', '', cleaned)
                                cleaned = cleaned.strip()
                                
                                if cleaned:
                                    artifacts[key] = cleaned
                                else:
                                    artifacts[key] = result
                            else:
                                # For primitive types (including string results from call_llm), store directly
                                artifacts[key] = result
                        else:
                            # For other types, convert to string
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
            
            # Check if plan is complete or if more tasks are needed
            logger.info("Checking if execution is complete...")
            
            # If plan has 0 tasks, it's considered complete (simple answer case)
            if len(all_tasks) == 0:
                logger.info("Plan has 0 tasks - considered complete for simple answer questions")
                break
            
            if page_data:
                completion_status = await check_plan_completion(plan, artifacts, page_data)
                logger.info(f"Completion status: {json.dumps(completion_status, indent=2)}")
                
                if completion_status.get("answer_ready"):
                    logger.info("Answer is ready! No more tasks needed.")
                    all_tasks = []  # Clear tasks to exit loop
                    break
                elif completion_status.get("needs_more_tasks") and iteration < max_iterations:
                    logger.info(f"More tasks needed. Generating next batch (iteration {iteration + 1})")
                    next_tasks = await generate_next_tasks(plan, artifacts, page_data, completion_status)
                    
                    if next_tasks:
                        # Add task IDs if missing
                        for i, task in enumerate(next_tasks):
                            if "id" not in task:
                                task["id"] = f"task_{iteration}_{i}"
                        all_tasks = next_tasks
                        logger.info(f"Generated {len(next_tasks)} next tasks")
                    else:
                        logger.info("No more tasks generated. Proceeding to submission.")
                        break
                else:
                    logger.info("Cannot determine next steps or max iterations reached.")
                    break
            else:
                logger.info("No page_data available. Skipping completion check.")
                break
        # Generate final answer - only extract from artifact when confirmed ready
        final_spec = plan.get("final_answer_spec", {})
        final_answer = None
        
        logger.info("Preparing final answer for submission")
        
        # Get the answer from the specified artifact or literal value
        if final_spec.get("from"):
            from_key = final_spec["from"]
            
            # Clean up from_key: remove quotes and 'static value' prefix if present
            import re
            cleaned_key = from_key.strip()
            cleaned_key = re.sub(r"^static value\s*['\"]?|['\"]?$", "", cleaned_key)
            cleaned_key = re.sub(r"^['\"]|['\"]$", "", cleaned_key).strip()
            
            # Check if from_key is a real artifact
            if from_key in artifacts:
                final_answer = artifacts[from_key]
                logger.info(f"Final answer extracted from artifact '{from_key}'")
            elif cleaned_key in artifacts:
                final_answer = artifacts[cleaned_key]
                logger.info(f"Final answer extracted from artifact '{cleaned_key}'")
            else:
                # If no artifact found, treat "from" as a literal instruction/value
                # This handles cases like "from": "anything you want" or "from": "any string"
                logger.info(f"Artifact '{from_key}' not found, using as literal answer: {cleaned_key}")
                final_answer = cleaned_key
        
        # If still no answer but plan has no tasks and is complete, provide a default
        if not final_answer and len(all_tasks) == 0:
            logger.info("Plan has no tasks and no artifact reference, providing default answer")
            # Check if there's a type hint
            answer_type = final_spec.get("type", "string")
            if answer_type == "boolean":
                final_answer = "true"
            elif answer_type == "number":
                final_answer = "0"
            elif answer_type == "json":
                final_answer = "{}"
            else:  # string or default
                final_answer = "completed"
        
        # Prepare and execute final submission
        submit_url = plan.get("submit_url")
        request_body_spec = plan.get("request_body", {})
        submission_result = None
        
        logger.info(f"[SUBMISSION_PREP] Submit URL: {submit_url}")
        logger.info(f"[SUBMISSION_PREP] Request body spec: {request_body_spec}")
        logger.info(f"[SUBMISSION_PREP] Email: {email}")
        logger.info(f"[SUBMISSION_PREP] Origin URL: {origin_url}")
        
        if submit_url and request_body_spec:
            # Build submission body with defaults for common key names
            submission_body = {}
            
            # Email field
            email_key = request_body_spec.get("email_key", "email")
            submission_body[email_key] = email
            logger.info(f"[SUBMISSION_BUILD] Email key: {email_key}")
            
            # Secret field
            secret_key = request_body_spec.get("secret_key", "secret")
            submission_body[secret_key] = SECRET
            logger.info(f"[SUBMISSION_BUILD] Secret key: {secret_key}")
            
            # URL field - try to use standard names if LLM provided a non-standard one
            url_key = request_body_spec.get("url_value", "url")
            # If the LLM provided something that looks like a placeholder (e.g., "origin_url"), 
            # convert it to the standard "url" key
            if url_key in ["origin_url", "target_url", "url_value"]:
                logger.info(f"[SUBMISSION_BUILD] URL key converted from '{url_key}' to 'url'")
                url_key = "url"
            # Always use the original quiz URL (origin_url parameter), not the plan's origin_url
            submission_body[url_key] = origin_url
            logger.info(f"[SUBMISSION_BUILD] URL key: {url_key}")
            
            logger.info(f"[SUBMISSION_PREP] Submission body before answer: {json.dumps(submission_body)[:300]}")
            
            # Add final answer if specified
            answer_key = request_body_spec.get("answer_key", "answer")
            if final_answer:
                submission_body[answer_key] = final_answer
                logger.info(f"[SUBMISSION_BUILD] Answer key: {answer_key}")
                logger.info(f"[SUBMISSION_BUILD] Final answer: {str(final_answer)[:200]}...")
            
            logger.info(f"[SUBMISSION_FINAL] Complete submission body: {json.dumps(submission_body)[:500]}")
            submission_result = await answer_submit(submit_url, submission_body)
            
            # Track attempt data if we have an attempt tracker
            if quiz_attempt:
                quiz_attempt.answer = final_answer
                quiz_attempt.submission_response = submission_result.get("response", {}) if submission_result else None
                quiz_attempt.correct = submission_result.get("response", {}).get("correct") if submission_result else None
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
    """Enhanced pipeline with iterative quiz solving and retry logic for wrong answers"""
    try:
        current_url = url
        quiz_chain = []
        quiz_runs = {}  # Track all attempts for each quiz URL
        max_chain_iterations = 10  # Prevent infinite loops
        
        while len(quiz_chain) < max_chain_iterations:
            logger.info(f"=== Processing Quiz: {current_url} ===")
            
            # Get or create quiz run tracker for this URL
            if current_url not in quiz_runs:
                quiz_runs[current_url] = QuizRun(current_url)
            
            quiz_run = quiz_runs[current_url]
            
            # Start a new attempt
            quiz_attempt = quiz_run.start_attempt()
            
            # Render page
            logger.info(f"[QUIZ_RUN] Rendering page for attempt {quiz_attempt.attempt_number}")
            page_data = await render_page(current_url)
            
            # Generate execution plan
            logger.info(f"[QUIZ_RUN] Generating execution plan for attempt {quiz_attempt.attempt_number}")
            plan_json = await call_llm_for_plan(page_data)
            
            # Parse and validate plan
            try:
                plan_obj = json.loads(plan_json[7:-4])
                logger.info(f"[QUIZ_RUN] Plan parsed: {len(plan_obj.get('tasks', []))} tasks")
            except json.JSONDecodeError as e:
                logger.error(f"[QUIZ_RUN] Invalid JSON plan: {e}")
                quiz_attempt.error = f"Invalid plan JSON: {e}"
                quiz_attempt.finish()
                return {"success": False, "error": f"Invalid plan JSON: {e}", "quiz_runs": {url: qr.to_dict() for url, qr in quiz_runs.items()}}
            
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
            response_data = submission_response.get("response", {})
            
            logger.info(f"[QUIZ_RESPONSE] Response: {json.dumps(response_data, indent=2)[:300]}")
            
            # Check if answer was correct
            is_correct = response_data.get("correct", False)
            logger.info(f"[QUIZ_RESPONSE] Answer correct: {is_correct}")
            
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
                
                logger.info(f"[QUIZ_RETRY] Answer was incorrect. Elapsed time: {elapsed_time:.1f}s / Max: {max_retry_time}s")
                
                if quiz_run.can_retry(max_retry_time):
                    logger.info(f"[QUIZ_RETRY] Can retry! Attempting again...")
                    # Loop will continue with next attempt
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
        
        return {
            "success": True,
            "quiz_chain": quiz_chain,
            "quiz_runs": {url: qr.to_dict() for url, qr in quiz_runs.items()},
            "total_quizzes_solved": len(quiz_chain),
            "final_result": quiz_chain[-1] if quiz_chain else None

        }
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

class VerifyPayload(BaseModel):
    email: EmailStr
    secret: str
    url: HttpUrl
    # Accept other arbitrary fields without validation errors
    # Store additional fields if needed
    class Config:
        extra = "allow"

@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "message": "Invalid JSON payload",
            "details": exc.errors(),
        },
    )

# ===== ENHANCED API ENDPOINT =====

@app.post("/solve")
async def verify(payload: VerifyPayload):
    """Enhanced solve endpoint with execution results"""
    logger.info(f"[API_RECEIVED] POST /solve")
    logger.info(f"[API_RECEIVED_PAYLOAD] Email: {payload.email}, URL: {str(payload.url)}")
    
    # Basic required checks
    if not payload.secret or not isinstance(payload.secret, str):
        logger.error(f"[API_ERROR] Invalid secret format")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if payload.secret != SECRET:
        logger.error(f"[API_ERROR] Invalid secret value")
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    logger.info(f"[API_AUTH] Authentication successful for {payload.email}")
    
    # Start pipeline with timing
    logger.info(f"[PIPELINE_START] Beginning quiz solving pipeline")
    started = time.time()
    pipeline_result = await run_pipeline(payload.email, str(payload.url))
    duration = time.time() - started
    logger.info(f"[PIPELINE_END] Pipeline completed in {duration:.2f}s - Success: {pipeline_result.get('success')}")
    
    # Prepare response
    response_data = {
        "status": "ok" if pipeline_result.get("success") else "error",
        "email": payload.email,
        "execution_time": f"{duration:.2f}s",
        "pipeline_result": pipeline_result
    }
    
    logger.info(f"[API_RESPONSE] Status: {response_data['status']}, Duration: {response_data['execution_time']}")
    logger.info(f"[API_RESPONSE_BODY] {json.dumps(response_data, indent=2)[:500]}...")
    
    # Remove sensitive data from response
    if "pipeline_result" in response_data and "artifacts" in response_data["pipeline_result"]:
        del response_data["pipeline_result"]["artifacts"]
    
    return JSONResponse(status_code=200, content=response_data)

