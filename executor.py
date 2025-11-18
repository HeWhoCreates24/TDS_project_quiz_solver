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
        
        # Extract multimedia sources (audio, video, images)
        audio_sources = await page.eval_on_selector_all("audio[src], audio source[src]", "els => els.map(e => e.src || e.getAttribute('src'))")
        video_sources = await page.eval_on_selector_all("video[src], video source[src]", "els => els.map(e => e.src || e.getAttribute('src'))")
        image_sources = await page.eval_on_selector_all("img[src]", "els => els.map(e => e.src)")
        
        # Try to extract any dynamically rendered content from divs with IDs
        rendered_divs = await page.eval_on_selector_all("div[id]", "els => els.map(e => ({id: e.id, text: e.innerText, html: e.innerHTML}))")
        
        await browser.close()
        return {
            "html": content,
            "text": text,
            "links": links,
            "code_blocks": pres,
            "rendered_divs": rendered_divs,
            "audio_sources": audio_sources,
            "video_sources": video_sources,
            "image_sources": image_sources
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

model = "openai/gpt-5-nano"

async def call_llm(prompt: str, system_prompt: str = None, max_tokens: int = 2000, temperature: float = 0) -> str:
    """Call LLM with given prompt"""
    if system_prompt is None:
        system_prompt = "You are a helpful assistant."
    
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

def get_tool_definitions() -> List[Dict[str, Any]]:
    """Define available tools for OpenAI function calling"""
    return [
        # SCRAPING TOOLS
        {
            "type": "function",
            "function": {
                "name": "render_js_page",
                "description": "Render a webpage with JavaScript execution and extract content including text, links, code blocks, and rendered div elements",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the page to render"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_text",
                "description": "Fetch text content from a URL via HTTP GET request",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch content from"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_from_api",
                "description": "Fetch data from API with custom headers and body (supports GET, POST, PUT, DELETE)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "API endpoint URL"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE"],
                            "description": "HTTP method"
                        },
                        "headers": {
                            "type": "object",
                            "description": "Optional custom headers"
                        },
                        "body": {
                            "type": "object",
                            "description": "Optional request body for POST/PUT"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        # DATA PARSING TOOLS
        {
            "type": "function",
            "function": {
                "name": "parse_csv",
                "description": "Parse a CSV file from a local path or URL and load it into a pandas DataFrame",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to the CSV file"
                        },
                        "path": {
                            "type": "string",
                            "description": "Local file path to the CSV file"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "parse_excel",
                "description": "Parse Excel file (.xlsx, .xls) into a DataFrame",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to Excel file"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "parse_json_file",
                "description": "Parse JSON file into structured data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to JSON file"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "parse_html_tables",
                "description": "Extract tables from HTML content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path_or_html": {
                            "type": "string",
                            "description": "HTML file path or raw HTML string"
                        }
                    },
                    "required": ["path_or_html"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "parse_pdf_tables",
                "description": "Extract tables from PDF files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to PDF file"
                        },
                        "pages": {
                            "type": "string",
                            "description": "Pages to extract (e.g., 'all', '1-3', '1,4,6')"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        # DATA CLEANSING TOOLS
        {
            "type": "function",
            "function": {
                "name": "clean_text",
                "description": "Clean and normalize text data (remove whitespace, special chars, normalize case)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to clean"
                        },
                        "remove_special_chars": {
                            "type": "boolean",
                            "description": "Remove special characters"
                        },
                        "normalize_whitespace": {
                            "type": "boolean",
                            "description": "Normalize whitespace"
                        }
                    },
                    "required": ["text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "extract_patterns",
                "description": "Extract patterns from text using regex (emails, URLs, phone numbers, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to extract from"
                        },
                        "pattern_type": {
                            "type": "string",
                            "enum": ["email", "url", "phone", "date", "number", "custom"],
                            "description": "Type of pattern to extract"
                        },
                        "custom_pattern": {
                            "type": "string",
                            "description": "Custom regex pattern if pattern_type is 'custom'"
                        }
                    },
                    "required": ["text", "pattern_type"]
                }
            }
        },
        # DATA PROCESSING TOOLS
        {
            "type": "function",
            "function": {
                "name": "transform_data",
                "description": "Transform data using various operations (reshape, pivot, melt, transpose)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "type": "string",
                            "description": "DataFrame key from registry"
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["pivot", "melt", "transpose", "reshape"],
                            "description": "Transformation operation"
                        },
                        "params": {
                            "type": "object",
                            "description": "Operation-specific parameters"
                        }
                    },
                    "required": ["dataframe", "operation"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "transcribe_audio",
                "description": "Transcribe audio file to text using speech recognition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to audio file"
                        }
                    },
                    "required": ["audio_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_image",
                "description": "Analyze image content using vision AI",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to image file"
                        },
                        "task": {
                            "type": "string",
                            "enum": ["describe", "ocr", "detect_objects", "classify"],
                            "description": "Analysis task to perform"
                        }
                    },
                    "required": ["image_path", "task"]
                }
            }
        },
        # DATA ANALYSIS TOOLS
        {
            "type": "function",
            "function": {
                "name": "dataframe_ops",
                "description": "Perform operations on pandas DataFrames such as filter, aggregate, transform, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Operation to perform (e.g., 'filter', 'aggregate', 'groupby', 'sort', 'select')",
                            "enum": ["filter", "aggregate", "groupby", "sort", "select", "transform", "count"]
                        },
                        "params": {
                            "type": "object",
                            "description": "Parameters for the operation including dataframe reference and operation-specific arguments"
                        }
                    },
                    "required": ["op", "params"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "filter_data",
                "description": "Filter DataFrame rows based on conditions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "type": "string",
                            "description": "DataFrame key from registry"
                        },
                        "filters": {
                            "type": "object",
                            "description": "Filter conditions (column: value pairs or expressions)"
                        }
                    },
                    "required": ["dataframe", "filters"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "sort_data",
                "description": "Sort DataFrame by one or more columns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "type": "string",
                            "description": "DataFrame key from registry"
                        },
                        "sort_by": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Column names to sort by"
                        },
                        "ascending": {
                            "type": "boolean",
                            "description": "Sort in ascending order"
                        }
                    },
                    "required": ["dataframe", "sort_by"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_statistics",
                "description": "Calculate statistical measures (mean, median, std, percentiles, sum, etc.) from a dataframe",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "type": "string",
                            "description": "DataFrame key from registry (e.g., 'df_0')"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to analyze (optional, defaults to all numeric columns)"
                        },
                        "stats": {
                            "type": "array",
                            "items": {"type": "string"},
                            "enum": ["mean", "median", "std", "min", "max", "count", "sum"],
                            "description": "Statistics to calculate (e.g., ['sum'], ['mean', 'std'])"
                        }
                    },
                    "required": ["dataframe", "stats"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "apply_ml_model",
                "description": "Apply machine learning models (regression, classification, clustering)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "type": "string",
                            "description": "DataFrame key from registry"
                        },
                        "model_type": {
                            "type": "string",
                            "enum": ["linear_regression", "logistic_regression", "kmeans", "decision_tree"],
                            "description": "ML model to apply"
                        },
                        "kwargs": {
                            "type": "object",
                            "description": "Model-specific parameters"
                        }
                    },
                    "required": ["dataframe", "model_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "geospatial_analysis",
                "description": "Perform geospatial analysis (distance calculation, geocoding, spatial joins)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "type": "string",
                            "description": "DataFrame key with geospatial data"
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["distance", "geocode", "spatial_join", "buffer"],
                            "description": "Type of geospatial analysis"
                        },
                        "kwargs": {
                            "type": "object",
                            "description": "Analysis-specific parameters"
                        }
                    },
                    "required": ["analysis_type"]
                }
            }
        },
        # VISUALIZATION TOOLS
        {
            "type": "function",
            "function": {
                "name": "create_chart",
                "description": "Create static charts (bar, line, scatter, pie, histogram, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "type": "string",
                            "description": "DataFrame key from registry"
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "scatter", "pie", "histogram", "box"],
                            "description": "Type of chart to create"
                        },
                        "x_col": {
                            "type": "string",
                            "description": "X-axis column name"
                        },
                        "y_col": {
                            "type": "string",
                            "description": "Y-axis column name"
                        },
                        "title": {
                            "type": "string",
                            "description": "Chart title"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to save chart image"
                        }
                    },
                    "required": ["dataframe", "chart_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_interactive_chart",
                "description": "Create interactive charts using Plotly (supports zoom, hover, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "type": "string",
                            "description": "DataFrame key from registry"
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "scatter", "pie", "3d_scatter"],
                            "description": "Type of interactive chart"
                        },
                        "x_col": {
                            "type": "string",
                            "description": "X-axis column"
                        },
                        "y_col": {
                            "type": "string",
                            "description": "Y-axis column"
                        },
                        "title": {
                            "type": "string",
                            "description": "Chart title"
                        }
                    },
                    "required": ["dataframe", "chart_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_narrative",
                "description": "Generate natural language narrative from data analysis results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe": {
                            "type": "string",
                            "description": "DataFrame key from registry"
                        },
                        "summary_stats": {
                            "type": "object",
                            "description": "Summary statistics to include in narrative"
                        }
                    },
                    "required": ["dataframe"]
                }
            }
        },
        # UTILITY TOOLS
        {
            "type": "function",
            "function": {
                "name": "call_llm",
                "description": "Call an LLM to analyze data, extract information, or generate content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to send to the LLM"
                        },
                        "system_prompt": {
                            "type": "string",
                            "description": "Optional system prompt to guide the LLM's behavior"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens in the response"
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Temperature for response generation (0.0 to 2.0)"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "download_file",
                "description": "Download a binary file from a URL (images, audio, video, documents, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the file to download"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "extract_audio_metadata",
                "description": "Extract metadata from audio file (duration, sample rate, channels, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to downloaded audio file"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "transcribe_audio",
                "description": "Transcribe audio file to text using speech-to-text. Use this to extract instructions or information from audio files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to transcribe"
                        }
                    },
                    "required": ["audio_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "make_plot",
                "description": "Create custom plots with detailed specifications",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spec": {
                            "type": "object",
                            "description": "Plot specification including type, data, styling"
                        }
                    },
                    "required": ["spec"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "zip_base64",
                "description": "Create a zip archive and encode as base64",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File paths to include in zip"
                        }
                    },
                    "required": ["paths"]
                }
            }
        }
    ]

async def call_llm_with_tools(page_data: Dict[str, Any], previous_attempts: List["QuizAttempt"] = None) -> Dict[str, Any]:
    """
    Generate execution plan using OpenAI function/tool calling API.
    This is more robust than prompt-based JSON generation.
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
                reason = prev.submission_response.get('reason', 'N/A')
                previous_context += f"  - Reason for failure: {reason}\n"
        previous_context += "\nUSE THIS INFORMATION TO GENERATE A BETTER PLAN!\n"
    
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
   - If audio says "min/minimum": calculate_statistics with {{"dataframe": "df_0", "stats": ["min"]}}
''' if csv_link else ''}

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
        "tool_choice": tool_choice_mode,  # Force tools when CSV detected
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
                            extracted_url = re.sub(r'[\'"}]+.*$', '', extracted_url)
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

async def check_plan_completion(plan_obj: Dict[str, Any], artifacts: Dict[str, Any], page_data: Dict[str, Any]) -> Dict[str, Any]:
    """Check if plan execution is complete"""
    
    # Quick check: if we have statistics results with actual values, we're done
    for artifact_key, artifact_value in artifacts.items():
        if isinstance(artifact_value, dict) and 'statistics' in artifact_value:
            stats_dict = artifact_value.get('statistics', {})
            if isinstance(stats_dict, dict):
                for col_stats in stats_dict.values():
                    if isinstance(col_stats, dict) and len(col_stats) > 0:
                        # We have actual statistics values - answer is ready
                        logger.info(f"[PLAN_COMPLETE] Found statistics in {artifact_key}: {col_stats}")
                        return {
                            "answer_ready": True,
                            "needs_more_tasks": False,
                            "reason": "Statistics calculated successfully",
                            "recommended_next_action": "submit"
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
    """Generate next batch of tasks using LLM with function calling"""
    
    # Build context about what we have and what we need
    artifacts_summary = {}
    for key, value in artifacts.items():
        if isinstance(value, str):
            artifacts_summary[key] = value[:200] if len(value) > 200 else value
        elif isinstance(value, dict):
            artifacts_summary[key] = {k: str(v)[:100] for k, v in list(value.items())[:5]}
        else:
            artifacts_summary[key] = str(value)[:200]
    
    system_prompt = """You are planning the NEXT steps to complete a quiz task.
    
You have already executed some tasks and produced artifacts. Now determine what additional steps are needed.

CRITICAL: Use function calling to specify the exact tools needed. Common patterns:

1. If you have downloaded an AUDIO file (content_type: audio/*):
   - MUST call transcribe_audio to convert speech to text
   - The transcribed text will contain instructions for what to do next
   
2. If you have transcribed audio text + CSV data:
   - Follow the instructions in the transcribed text
   - Use calculate_statistics or dataframe_ops based on what the audio says

3. If you have dataframe metadata but need to analyze it (sum, count, filter):
   - Call calculate_statistics or dataframe_ops on the dataframe
   
4. If you have HTML/text but need to extract specific information:
   - Call call_llm with a prompt to extract the needed information

5. If you need more data from URLs:
   - Call render_js_page or fetch_text

Use the dataframe_key from parse_csv results to operate on the actual data."""

    prompt = f"""
CURRENT STATE:
Artifacts produced so far: {json.dumps(artifacts_summary, indent=2)}

Analysis: {json.dumps(completion_status, indent=2)}

Quiz page text: {page_data.get('text', 'N/A')[:500]}
Quiz page links: {page_data.get('links', [])}
Quiz page audio sources: {page_data.get('audio_sources', [])}

TASK: What additional tool calls are needed to get the final answer?

IMPORTANT CHECKS:
1. Do you see a downloaded audio file (with "path" and "content_type": "audio/...")? → Call transcribe_audio on that path
2. Do you see transcribed audio text? → Read it to determine what analysis to perform
3. Do you see a CSV link in the page? → Call parse_csv to load it
4. Do you see dataframe metadata with a 'dataframe_key'? → Call calculate_statistics to analyze it based on the instructions

The final answer is NOT file metadata - it's the result of analyzing data based on audio instructions.
"""

    OPEN_AI_BASE_URL = os.getenv("LLM_BASE_URL", "https://aipipe.org/openrouter/v1/chat/completions")
    API_KEY = os.getenv("API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Get tool definitions (reuse from call_llm_with_tools)
    tools = get_tool_definitions()
    
    json_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 2000,
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OPEN_AI_BASE_URL, headers=headers, json=json_data, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            message = data["choices"][0]["message"]
            
            # Check if model wants to call tools
            if message.get("tool_calls"):
                logger.info(f"[NEXT_TASKS] Model requested {len(message['tool_calls'])} additional tool call(s)")
                
                # Convert tool calls to task format
                next_tasks = []
                for i, tool_call in enumerate(message["tool_calls"]):
                    func_name = tool_call["function"]["name"]
                    func_args = json.loads(tool_call["function"]["arguments"])
                    
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
                logger.info(f"[NEXT_TASKS] No additional tasks needed")
                return []
                
    except Exception as e:
        logger.error(f"[NEXT_TASKS] Error generating tasks: {e}")
        return []

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
                    elif tool_name == "extract_audio_metadata":
                        from tools import MultimediaTools
                        logger.info(f"[TOOL_EXEC] {task_id}: extract_audio_metadata - Path: {inputs.get('audio_path') or inputs.get('path')}")
                        audio_path = inputs.get("audio_path") or inputs.get("path")
                        result = MultimediaTools.extract_audio_metadata(audio_path)
                        logger.info(f"[TOOL_RESULT] {task_id}: extract_audio_metadata - {result}")
                    elif tool_name == "transcribe_audio":
                        from tools import MultimediaTools
                        logger.info(f"[TOOL_EXEC] {task_id}: transcribe_audio - Path: {inputs.get('audio_path')}")
                        audio_path = inputs.get("audio_path")
                        result = await MultimediaTools.transcribe_audio(audio_path)
                        logger.info(f"[TOOL_RESULT] {task_id}: transcribe_audio - Text: {result.get('text', '')[:200]}...")
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
                        df_key = inputs["dataframe"]
                        if df_key not in dataframe_registry:
                            raise ValueError(f"DataFrame '{df_key}' not found in registry. Available: {list(dataframe_registry.keys())}")
                        df = dataframe_registry[df_key]
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
                                if hasattr(final_answer, 'item'):
                                    final_answer = final_answer.item()
                                logger.info(f"[ARTIFACT_EXTRACTION] Extracted first stat: {final_answer}")
                    else:
                        logger.warning(f"[ARTIFACT_EXTRACTION] Empty statistics dict for column {first_col}")
            # Actual dict object with 'text' key
            elif 'text' in final_answer:
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
            
            # Generate execution plan using function calling
            logger.info(f"[QUIZ_RUN] Generating execution plan for attempt {quiz_attempt.attempt_number}")
            llm_response = await call_llm_with_tools(page_data, quiz_run.attempts[:-1])  # Pass previous attempts
            logger.info(f"[LLM_TOOLS_RESPONSE] Tool calls: {llm_response.get('tool_calls') is not None}, Content: {str(llm_response.get('content', ''))[:200]}")
            
            # Build plan object from LLM response
            plan_obj = {
                "submit_url": "https://tds-llm-analysis.s-anand.net/submit",  # Default, should be extracted from page
                "origin_url": current_url,
                "tasks": llm_response.get("tasks", []),
                "final_answer_spec": {
                    "type": "string",
                    "from": None  # Will be determined based on tasks or content
                },
                "request_body": {
                    "email_key": "email",
                    "secret_key": "secret",
                    "url_value": "url",
                    "answer_key": "answer"
                }
            }
            
            # If tasks exist, set final_answer_spec to reference the last task's output
            if plan_obj["tasks"]:
                last_task = plan_obj["tasks"][-1]
                if last_task.get("produces"):
                    # Reference the key from the last task's produces
                    output_key = last_task["produces"][0]["key"]
                    plan_obj["final_answer_spec"]["from"] = output_key
                    logger.info(f"[QUIZ_PLAN] Set final answer to reference artifact: {output_key}")
            
            # If no tasks and model gave direct content, use that as answer
            if not plan_obj["tasks"] and llm_response.get("content"):
                logger.info(f"[QUIZ_PLAN] Direct answer from LLM: {str(llm_response['content'])[:100]}")
                plan_obj["final_answer_spec"]["from"] = str(llm_response["content"])
            
            # FALLBACK: If LLM didn't call tools but there are links, force smart scraping
            if not plan_obj["tasks"] and page_data.get("links"):
                logger.warning(f"[QUIZ_PLAN] LLM did not call tools but page has {len(page_data['links'])} links. Forcing smart scraping.")
                for i, link in enumerate(page_data['links'][:3]):  # Limit to first 3 links
                    if link and link.startswith('http'):
                        # Check file extension to choose appropriate tool
                        link_lower = link.lower()
                        if '.csv' in link_lower:
                            tool_name = "parse_csv"
                            result_key = f"csv_data_{i+1}"
                            logger.info(f"[QUIZ_PLAN] Forcing CSV parse: {link}")
                        elif '.xlsx' in link_lower or '.xls' in link_lower:
                            tool_name = "parse_excel"
                            result_key = f"excel_data_{i+1}"
                            logger.info(f"[QUIZ_PLAN] Forcing Excel parse: {link}")
                        elif '.json' in link_lower:
                            tool_name = "parse_json_file"
                            result_key = f"json_data_{i+1}"
                            logger.info(f"[QUIZ_PLAN] Forcing JSON parse: {link}")
                        else:
                            tool_name = "render_js_page"
                            result_key = f"scraped_content_{i+1}"
                            logger.info(f"[QUIZ_PLAN] Forcing page render: {link}")
                        
                        plan_obj["tasks"].append({
                            "id": f"forced_task_{i+1}",
                            "tool_name": tool_name,
                            "inputs": {"url": link} if tool_name == "parse_csv" else {"url": link} if tool_name == "render_js_page" else {"path": link},
                            "produces": [{"key": result_key, "type": "json"}],
                            "notes": f"Forced {tool_name} of link: {link}"
                        })
                
                # Update final answer to reference last result
                if plan_obj["tasks"]:
                    plan_obj["final_answer_spec"]["from"] = plan_obj["tasks"][-1]["produces"][0]["key"]
                
            # ADDITIONAL FALLBACK: If we parsed data files AND page explicitly asks for analysis
            # This runs AFTER the link scraping fallback above
            page_text_lower = page_data.get("text", "").lower()
            data_tasks = [t for t in plan_obj["tasks"] if t["tool_name"] in ["parse_csv", "parse_excel", "parse_json_file"]]
            
            # Only add analysis if the page EXPLICITLY mentions analysis operations
            analysis_keywords = ["sum", "total", "count", "average", "mean", "filter", "calculate", "add up", "compute"]
            needs_analysis = any(keyword in page_text_lower for keyword in analysis_keywords)
            
            logger.info(f"[FALLBACK_CHECK] Found {len(data_tasks)} data parsing tasks")
            
            # If we parsed data AND page asks for analysis, add analysis step
            if data_tasks and needs_analysis:
                logger.info(f"[QUIZ_PLAN] Data file parsed. Adding analysis task (raw dataframe is never the answer).")
                
                # Determine which analysis to add based on keywords
                if "sum" in page_text_lower or "total" in page_text_lower or "add" in page_text_lower:
                    stats_to_compute = ["sum"]
                    analysis_type = "sum"
                elif "count" in page_text_lower:
                    stats_to_compute = ["count"]
                    analysis_type = "count"
                elif "average" in page_text_lower or "mean" in page_text_lower:
                    stats_to_compute = ["mean"]
                    analysis_type = "mean"
                else:
                    # Default: compute sum (most common for numeric data)
                    stats_to_compute = ["sum"]
                    analysis_type = "sum (default)"
                
                analysis_task = {
                    "id": f"forced_analysis_1",
                    "tool_name": "calculate_statistics",
                    "inputs": {
                        "dataframe": "df_0",  # DataFrame key from registry
                        "columns": [],  # Empty means all columns
                        "stats": stats_to_compute
                    },
                    "produces": [{"key": "analysis_result_1", "type": "json"}],
                    "notes": f"Forced analysis: calculate {analysis_type} of data"
                }
                
                plan_obj["tasks"].append(analysis_task)
                plan_obj["final_answer_spec"]["from"] = "analysis_result_1"
                logger.info(f"[QUIZ_PLAN] Added forced analysis task: calculate {analysis_type}")
            
            # Try to extract submit URL from page data
            if "submit" in page_data.get("text", "").lower():
                submit_match = re.search(r'(https?://[^\s]+/submit[^\s]*)', page_data.get("text", ""))
                if submit_match:
                    plan_obj["submit_url"] = submit_match.group(1)
                    logger.info(f"[QUIZ_PLAN] Extracted submit URL: {plan_obj['submit_url']}")
            
            tasks_count = len(plan_obj.get("tasks", []))
            logger.info(f"[QUIZ_RUN] Plan created with {tasks_count} tasks")
            logger.info(f"[QUIZ_PLAN] Full plan: {json.dumps(plan_obj, indent=2)[:1000]}")
            
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
