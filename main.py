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
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=body, timeout=30)
            response.raise_for_status()
            return {
                "status_code": response.status_code,
                "response": response.json() if response.content else {},
                "headers": dict(response.headers)
            }
    except Exception as e:
        logger.error(f"Error submitting answer to {url}: {e}")
        logger.error(f"Response body: {response.text if 'response' in locals() else 'N/A'}")
        print(f"Submission body: {body}")
        raise

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

        0 answer_submit(url, body): POST JSON to submit URL.
        1 render_js_page(url): Render JS, return {html, text, links, code_blocks}.
        2 fetch_text(url): GET, return text and headers.
        3 download_file(url): GET binary, return {path, content_type, size}.
        4 parse_csv(path), parse_excel(path), parse_json_file(path), parse_html_tables(path_or_html), parse_pdf_tables(path, pages).
        5 dataframe_ops(op, params): Perform operations like select/filter/sum/groupby on DataFrames registry keys.
        6 make_plot(spec): Return data URI base64 image.
        7 zip_base64(paths): Return data URI base64 zip.

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
    
# ===== PLAN EXECUTION ENGINE =====

async def execute_plan(plan_obj: Dict[str, Any], email: str, origin_url: str, page_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute the LLM-generated plan"""
    try:
        plan = plan_obj
        artifacts = {}
        execution_log = []
        
        logger.info(f"Starting execution of plan with {len(plan.get('tasks', []))} tasks")
        
        # Execute each task sequentially
        for task in plan.get("tasks", []):
            task_id = task["id"]
            tool_name = task["tool_name"]
            inputs = task.get("inputs", {})
            produces = task.get("produces", [])
            notes = task.get("notes", "")
            
            logger.info(f"Executing task {task_id}: {tool_name}")
            
            try:
                # Execute tool based on name
                if tool_name == "render_js_page":
                    result = await render_page(inputs["url"])
                elif tool_name == "fetch_text":
                    result = await fetch_text(inputs["url"])
                elif tool_name == "download_file":
                    result = await download_file(inputs["url"])
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
                elif tool_name == "answer_submit":
                    # For answer_submit during execution (not final submission)
                    result = await answer_submit(inputs["url"], inputs["body"])
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
                
                # Store produced artifacts
                for produce in produces:
                    key = produce["key"]
                    if isinstance(result, dict) and key in result:
                        artifacts[key] = result[key]
                    else:
                        artifacts[key] = result
                
                execution_log.append({
                    "task_id": task_id,
                    "status": "success",
                    "tool": tool_name,
                    "produces": [p["key"] for p in produces]
                })
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                execution_log.append({
                    "task_id": task_id,
                    "status": "failed",
                    "tool": tool_name,
                    "error": str(e)
                })
                raise
        
        # Generate final answer using LLM
        final_spec = plan.get("final_answer_spec", {})
        final_answer = None
        
        logger.info("Generating final answer with LLM")
        try:
            if page_data:
                # Call LLM to generate answer with access to artifacts
                final_answer = await call_llm_for_answer(plan_obj, artifacts, page_data)
            else:
                # Fallback: try to extract from artifacts directly
                if final_spec.get("from"):
                    from_key = final_spec["from"]
                    if from_key in artifacts:
                        final_answer = artifacts[from_key]
                    else:
                        final_answer = artifacts.get(from_key)
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            # Fallback to direct artifact lookup
            if final_spec.get("from"):
                from_key = final_spec["from"]
                final_answer = artifacts.get(from_key)
        
        # Prepare and execute final submission
        submit_url = plan.get("submit_url")
        request_body_spec = plan.get("request_body", {})
        submission_result = None
        
        logger.info(f"Submit URL: {submit_url}")
        logger.info(f"Request body spec: {request_body_spec}")
        logger.info(f"Email: {email}, Origin URL: {origin_url}")
        
        if submit_url and request_body_spec:
            # Build submission body with defaults for common key names
            submission_body = {}
            
            # Email field
            email_key = request_body_spec.get("email_key", "email")
            submission_body[email_key] = email
            
            # Secret field
            secret_key = request_body_spec.get("secret_key", "secret")
            submission_body[secret_key] = SECRET
            
            # URL field - try to use standard names if LLM provided a non-standard one
            url_key = request_body_spec.get("url_value", "url")
            # If the LLM provided something that looks like a placeholder (e.g., "origin_url"), 
            # convert it to the standard "url" key
            if url_key in ["origin_url", "target_url", "url_value"]:
                url_key = "url"
            submission_body[url_key] = plan.get("origin_url", origin_url)
            
            logger.info(f"Submission body before adding answer: {submission_body}")
            
            # Add final answer if specified
            answer_key = request_body_spec.get("answer_key", "answer")
            if final_answer:
                submission_body[answer_key] = final_answer
            
            logger.info(f"Final submission body: {submission_body}")
            submission_result = await answer_submit(submit_url, submission_body)
        
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
        return {
            "success": False,
            "error": str(e),
            "execution_log": execution_log
        }

async def run_pipeline(email: str, url: str) -> Dict[str, Any]:
    """Enhanced pipeline with plan execution"""
    try:
        # Render page
        logger.info(f"Rendering page: {url}")
        page_data = await render_page(url)
        
        # Generate execution plan
        logger.info("Generating execution plan with LLM")
        plan_json = await call_llm_for_plan(page_data)
        
        # Parse and validate plan
        try:
            plan_obj = json.loads(plan_json[7:-4])
            logger.info(f"Plan parsed successfully with {len(plan_obj.get('tasks', []))} tasks")
            logger.info(f"Full plan object: {json.dumps(plan_obj, indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON plan: {e}")
            print(plan_json[7:-4])
            return {"success": False, "error": f"Invalid plan JSON: {e}"}
        
        # Execute plan
        logger.info("Executing plan")
        execution_result = await execute_plan(plan_obj, email, url, page_data)
        
        return execution_result
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
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
    # Basic required checks
    if not payload.secret or not isinstance(payload.secret, str):
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if payload.secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    # Start pipeline with timing
    started = time.time()
    pipeline_result = await run_pipeline(payload.email, str(payload.url))
    duration = time.time() - started
    
    # Prepare response
    response_data = {
        "status": "ok" if pipeline_result.get("success") else "error",
        "email": payload.email,
        "execution_time": f"{duration:.2f}s",
        "pipeline_result": pipeline_result
    }
    
    # Remove sensitive data from response
    if "pipeline_result" in response_data and "artifacts" in response_data["pipeline_result"]:
        del response_data["pipeline_result"]["artifacts"]
    
    return JSONResponse(status_code=200, content=response_data)

