"""
Tool execution functions for quiz solving
Handles web scraping, file parsing, data operations, and artifact generation
"""
import os
import json
import re
import base64
import io
import zipfile
import logging
import threading
from typing import Any, Dict, List
from tempfile import NamedTemporaryFile
from playwright.async_api import async_playwright
import httpx
import pandas as pd
import matplotlib.pyplot as plt
from cache_manager import quiz_cache, hash_content

logger = logging.getLogger(__name__)

# Global registry for dataframes with thread-safe lock
dataframe_registry = {}
_dataframe_lock = threading.Lock()


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
    # Check cache first (5 min TTL for dynamic content)
    cached = quiz_cache.get_rendered_page(url, ttl=300)
    if cached:
        quiz_cache.record_time_saved(3.0)  # Rendering typically takes ~3s
        return cached
    
    # Cache miss - render page
    quiz_cache.record_miss()
    start_time = __import__('time').time()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        
        # Wait for any dynamic content to render (up to 3 seconds)
        try:
            await page.wait_for_timeout(3000)
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
        
        result = {
            "html": content,
            "text": text,
            "links": links,
            "code_blocks": pres,
            "rendered_divs": rendered_divs,
            "audio_sources": audio_sources,
            "video_sources": video_sources,
            "image_sources": image_sources
        }
        
        # Cache the result
        quiz_cache.set_rendered_page(url, result, ttl=300)
        return result


async def fetch_text(url: str) -> Dict[str, Any]:
    """Fetch text content from URL"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return {"content": response.text, "status_code": response.status_code}
    except Exception as e:
        logger.error(f"Error fetching text from {url}: {e}")
        raise


async def download_file(url: str) -> Dict[str, Any]:
    """Download file and return metadata"""
    # Check cache first (1 hour TTL)
    cached = quiz_cache.get_file(url, ttl=3600)
    if cached:
        quiz_cache.record_time_saved(2.0)  # Downloads typically take ~2s
        return cached
    
    # Cache miss - download file
    quiz_cache.record_miss()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=60.0)
            response.raise_for_status()
            
            # Save to temp file
            with NamedTemporaryFile(delete=False, suffix=os.path.splitext(url)[1]) as tmp:
                tmp.write(response.content)
                path = tmp.name
            
            result = {
                "path": path,
                "size": len(response.content),
                "content_type": response.headers.get("content-type")
            }
            
            # Cache the result
            quiz_cache.set_file(url, result, ttl=3600)
            return result
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {e}")
        raise


async def parse_csv_async(path: str = None, url: str = None) -> Dict[str, Any]:
    """Parse CSV file from path or URL (async version)"""
    try:
        # Determine source - handle case where URL is passed as path
        if not path and not url:
            raise ValueError("Either path or url must be provided")
        
        # Infrastructure: If path looks like URL, treat it as url
        if path and (path.startswith('http://') or path.startswith('https://')):
            url = path
            path = None
        
        # Cache miss - parse CSV
        quiz_cache.record_miss()
        
        # If URL, fetch content asynchronously
        if url:
            from io import StringIO
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0, follow_redirects=True)
                response.raise_for_status()
                source = StringIO(response.text)
        else:
            source = path
        
        # Try reading first to detect if it has headers
        sample_df = pd.read_csv(source, nrows=5)
        
        # Heuristic: If the first row (column names) are all numeric, likely no header
        has_header = True
        try:
            pd.to_numeric(sample_df.columns)
            has_header = False
        except:
            has_header = True
        
        # Reset source for full read if it's from URL
        if url:
            from io import StringIO
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0, follow_redirects=True)
                response.raise_for_status()
                source = StringIO(response.text)
        
        # Read the full CSV with appropriate header setting
        if has_header:
            df = pd.read_csv(source)
        else:
            df = pd.read_csv(source, header=None)
        
        # Thread-safe registry write
        with _dataframe_lock:
            df_key = f"df_{len(dataframe_registry)}"
            dataframe_registry[df_key] = df
        
        result = {
            "dataframe_key": df_key,
            "shape": df.shape,
            "columns": list(df.columns),
            "sample": df.head().to_dict()
        }
        
        return result
    except Exception as e:
        logger.error(f"Error parsing CSV from {url or path}: {e}")
        raise


def parse_csv(path: str = None, url: str = None) -> Dict[str, Any]:
    """Parse CSV file from path or URL (sync wrapper)"""
    import asyncio
    return asyncio.run(parse_csv_async(path, url))


async def parse_excel_async(path: str) -> Dict[str, Any]:
    """Parse Excel file (async version)"""
    import asyncio
    logger.info(f"[EXCEL_PARSE] Starting to parse {path}")
    try:
        # Handle URLs by downloading first
        if path.startswith('http://') or path.startswith('https://'):
            logger.info(f"[EXCEL_PARSE] Detected URL - downloading first")
            import urllib.request
            import tempfile
            loop = asyncio.get_event_loop()
            
            def _download():
                response = urllib.request.urlopen(path)
                content = response.read()
                # Create temp file with proper extension
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                temp_file.write(content)
                temp_file.close()
                return temp_file.name
            
            local_path = await loop.run_in_executor(None, _download)
            logger.info(f"[EXCEL_PARSE] Downloaded to {local_path}")
        else:
            local_path = path
        
        # Run blocking I/O in thread pool
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, pd.read_excel, local_path)
        logger.info(f"[EXCEL_PARSE] File loaded, shape: {df.shape}")
        
        # Thread-safe registry write
        with _dataframe_lock:
            df_key = f"df_{len(dataframe_registry)}"
            dataframe_registry[df_key] = df
        logger.info(f"[EXCEL_PARSE] Registered as {df_key}")
        
        return {
            "dataframe_key": df_key,
            "shape": df.shape,
            "columns": list(df.columns),
            "sample": df.head().to_dict()
        }
    except Exception as e:
        logger.error(f"Error parsing Excel {path}: {e}")
        raise


def parse_excel(path: str) -> Dict[str, Any]:
    """Parse Excel file (sync wrapper)"""
    import asyncio
    return asyncio.run(parse_excel_async(path))


async def parse_json_file_async(path: str) -> Dict[str, Any]:
    """Parse JSON file (async version)"""
    import asyncio
    logger.info(f"[JSON_PARSE] Starting to parse {path}")
    try:
        loop = asyncio.get_event_loop()
        
        def _read_json():
            with open(path, 'r') as f:
                return json.load(f)
        
        data = await loop.run_in_executor(None, _read_json)
        logger.info(f"[JSON_PARSE] Loaded {type(data).__name__} with {len(data) if isinstance(data, (list, dict)) else 'N/A'} items")
        return {"data": data, "type": type(data).__name__}
    except Exception as e:
        logger.error(f"Error parsing JSON {path}: {e}")
        raise


def parse_json_file(path: str) -> Dict[str, Any]:
    """Parse JSON file (sync wrapper)"""
    import asyncio
    return asyncio.run(parse_json_file_async(path))


async def parse_html_tables_async(html_content: str) -> Dict[str, Any]:
    """Parse HTML tables (async version)"""
    import asyncio
    logger.info(f"[HTML_PARSE] Starting to parse HTML (length: {len(html_content)})")
    try:
        loop = asyncio.get_event_loop()
        tables = await loop.run_in_executor(None, pd.read_html, html_content)
        logger.info(f"[HTML_PARSE] Found {len(tables)} table(s)")
        result = {}
        
        # Thread-safe registry writes
        with _dataframe_lock:
            for i, table in enumerate(tables):
                df_key = f"df_{len(dataframe_registry)}"
                dataframe_registry[df_key] = table
                result[f"table_{i}"] = {
                    "dataframe_key": df_key,
                    "shape": table.shape
                }
        logger.info(f"[HTML_PARSE] Registered {len(tables)} dataframes")
        return {"tables": result}
    except Exception as e:
        logger.error(f"Error parsing HTML tables: {e}")
        raise


def parse_html_tables(html_content: str) -> Dict[str, Any]:
    """Parse HTML tables (sync wrapper)"""
    import asyncio
    return asyncio.run(parse_html_tables_async(html_content))


async def parse_pdf_tables_async(path: str, pages: str = "all") -> Dict[str, Any]:
    """Parse PDF tables and store in dataframe registry (async version)"""
    import asyncio
    logger.info(f"[PDF_PARSE] Starting to parse {path}, pages: {pages}")
    try:
        import pdfplumber
        
        # Handle dict input from download_file artifact
        if isinstance(path, dict):
            path = path.get('path', path)
        
        # Run blocking I/O in thread pool
        loop = asyncio.get_event_loop()
        
        def _extract_pdf_tables():
            all_tables = []
            with pdfplumber.open(path) as pdf:
                page_list = range(len(pdf.pages)) if pages == "all" else [int(p) - 1 for p in pages.split(",")]
                
                for page_num in page_list:
                    if page_num < len(pdf.pages):
                        page = pdf.pages[page_num]
                        tables = page.extract_tables()
                        all_tables.extend(tables)
            return all_tables
        
        all_tables = await loop.run_in_executor(None, _extract_pdf_tables)
        logger.info(f"[PDF_PARSE] Extracted {len(all_tables)} table(s)")
        
        if not all_tables:
            logger.warning(f"No tables found in PDF {path}")
            return {"dataframe_key": None, "tables_found": 0}
        
        # Convert first table to dataframe
        df = pd.DataFrame(all_tables[0][1:], columns=all_tables[0][0])
        
        # Convert numeric columns from strings to numbers
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Keep as string if conversion fails
                pass
        
        df_key = f"df_{len(dataframe_registry)}"
        dataframe_registry[df_key] = df
        logger.info(f"[PDF_PARSE] Registered as {df_key}, shape: {df.shape}")
        
        logger.info(f"[PDF_PARSE] Extracted {len(all_tables)} table(s), registered as {df_key}")
        
        return {
            "dataframe_key": df_key,
            "tables_found": len(all_tables),
            "shape": df.shape,
            "columns": list(df.columns),
            "sample": df.head(3).to_dict()
        }
    except ImportError:
        logger.error("pdfplumber not installed - cannot parse PDF tables")
        raise ImportError("Please install pdfplumber: pip install pdfplumber")
    except Exception as e:
        logger.error(f"Error parsing PDF tables from {path}: {e}")
        raise


def parse_pdf_tables(path: str, pages: str = "all") -> Dict[str, Any]:
    """Parse PDF tables (sync wrapper)"""
    import asyncio
    return asyncio.run(parse_pdf_tables_async(path, pages))


def extract_patterns(text: str, pattern_type: str, custom_pattern: str = None) -> Dict[str, Any]:
    """
    Extract patterns from text using regex
    
    Args:
        text: Text to extract from (can be dict with 'content' key or raw string)
        pattern_type: Type of pattern - "email", "url", "phone", "date", "number", "custom"
        custom_pattern: Custom regex pattern if pattern_type is "custom"
    
    Returns:
        Dict with "matches" (list of found patterns) and "count" (unique count)
    """
    import re
    
    # Handle dict input (from fetch_text artifact)
    if isinstance(text, dict):
        text = text.get('content', str(text))
    
    text = str(text)
    
    # Define common patterns
    patterns = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "url": r'https?://[^\s]+',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        "number": r'\b\d+\.?\d*\b',
    }
    
    if pattern_type == "custom":
        if not custom_pattern:
            raise ValueError("custom_pattern required when pattern_type='custom'")
        pattern = custom_pattern
    elif pattern_type in patterns:
        pattern = patterns[pattern_type]
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}. Use: {list(patterns.keys())} or 'custom'")
    
    # Find all matches
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Get unique matches (preserve order)
    seen = set()
    unique_matches = []
    for match in matches:
        match_lower = match.lower()
        if match_lower not in seen:
            seen.add(match_lower)
            unique_matches.append(match)
    
    result = {
        "matches": unique_matches,
        "count": len(unique_matches),
        "pattern_type": pattern_type
    }
    
    logger.info(f"[TOOL_RESULT] extract_patterns - Found {len(unique_matches)} unique {pattern_type}(s)")
    return result


def dataframe_ops(op: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform DataFrame operations (thread-safe)"""
    df_key = params.get("dataframe_key")
    
    # Thread-safe registry read - copy to avoid race conditions during filtering
    with _dataframe_lock:
        if df_key not in dataframe_registry:
            raise ValueError(f"DataFrame {df_key} not found in registry. Available: {list(dataframe_registry.keys())}")
        df = dataframe_registry[df_key].copy()
    
    result = None
    
    if op == "select":
        columns = params.get("columns", [])
        result = df[columns] if columns else df
    elif op == "filter":
        condition = params.get("condition")
        if condition:
            try:
                # Handle dict format: {'column': 'name', 'operator': '>=', 'value': 100}
                if isinstance(condition, dict):
                    col = condition.get('column')
                    op_str = condition.get('operator')
                    val = condition.get('value')
                    
                    # Validate column exists - EXACT match required (case-sensitive)
                    if col not in df.columns:
                        raise ValueError(f"Column '{col}' not found. Available columns: {list(df.columns)}. Column names are CASE-SENSITIVE.")
                    
                    # Direct execution with dict format - more reliable
                    try:
                        value = pd.to_numeric(val)
                    except:
                        value = str(val).strip("'\"")
                    
                    if op_str == ">=":
                        result = df[df[col] >= value]
                    elif op_str == "<=":
                        result = df[df[col] <= value]
                    elif op_str == ">":
                        result = df[df[col] > value]
                    elif op_str == "<":
                        result = df[df[col] < value]
                    elif op_str == "==":
                        result = df[df[col] == value]
                    elif op_str == "!=":
                        result = df[df[col] != value]
                    else:
                        raise ValueError(f"Unsupported operator: {op_str}")
                else:
                    # Parse string condition: "column op value" or "value op column"
                    # Support both orders: "96903 >= 1371" and "column >= value"
                    parts = condition.split()
                    if len(parts) >= 3:
                        # Try parsing as "column op value"
                        col_name = parts[0]
                        operator = parts[1]
                        value_str = " ".join(parts[2:])
                        
                        # Check if first part is a column name (handle both string and int column names)
                        col_name_normalized = col_name
                        try:
                            # Try converting to int for numeric column names
                            col_name_normalized = int(col_name)
                        except (ValueError, TypeError):
                            pass
                        
                        if col_name_normalized in df.columns:
                            # Normal order: column op value
                            try:
                                value = pd.to_numeric(value_str)
                            except:
                                value = value_str.strip("'\"")
                            
                            if operator == ">=":
                                result = df[df[col_name_normalized] >= value]
                            elif operator == "<=":
                                result = df[df[col_name_normalized] <= value]
                            elif operator == ">":
                                result = df[df[col_name_normalized] > value]
                            elif operator == "<":
                                result = df[df[col_name_normalized] < value]
                            elif operator == "==":
                                result = df[df[col_name_normalized] == value]
                            elif operator == "!=":
                                result = df[df[col_name_normalized] != value]
                            else:
                                # Invalid operator
                                available_cols = list(df.columns)
                                raise ValueError(
                                    f"Invalid operator '{operator}' in condition: '{condition}'. "
                                    f"Supported operators: >=, <=, >, <, ==, !=. "
                                    f"Available columns (CASE-SENSITIVE): {available_cols}"
                                )
                        else:
                            # Reversed order: value op column
                            # Need to find the column (should be last part)
                            col_name = parts[-1]
                            value_str = parts[0]
                            
                            try:
                                value = pd.to_numeric(value_str)
                            except:
                                value = value_str.strip("'\"")
                            
                            # Reverse the operator
                            if operator == ">=":
                                result = df[df[col_name] <= value]  # a >= b means b <= a
                            elif operator == "<=":
                                result = df[df[col_name] >= value]
                            elif operator == ">":
                                result = df[df[col_name] < value]
                            elif operator == "<":
                                result = df[df[col_name] > value]
                            elif operator == "==":
                                result = df[df[col_name] == value]
                            elif operator == "!=":
                                result = df[df[col_name] != value]
                            else:
                                # Invalid operator in reversed condition
                                available_cols = list(df.columns)
                                raise ValueError(
                                    f"Invalid operator '{operator}' in condition: '{condition}'. "
                                    f"Supported operators: >=, <=, >, <, ==, !=. "
                                    f"Available columns (CASE-SENSITIVE): {available_cols}"
                                )
                    else:
                        # Provide helpful error for unsupported patterns
                        available_cols = list(df.columns)
                        raise ValueError(
                            f"Invalid condition format: '{condition}'. "
                            f"Supported format: 'ColumnName operator value' (e.g., 'ID >= 1'). "
                            f"Operators: >=, <=, >, <, ==, !=. "
                            f"Available columns (CASE-SENSITIVE): {available_cols}. "
                            f"Note: Column names must match exactly."
                        )
            except Exception as e:
                logger.error(f"Error evaluating condition '{condition}': {e}")
                # Re-raise with helpful context if not already detailed
                if "Invalid condition format" in str(e) or "Available columns" in str(e):
                    raise
                available_cols = list(df.columns)
                raise ValueError(
                    f"Filter condition error: '{condition}'. "
                    f"Error: {str(e)}. "
                    f"Available columns (CASE-SENSITIVE): {available_cols}. "
                    f"Use format: 'ColumnName operator value' with operators: >=, <=, >, <, ==, !="
                )
        else:
            result = df
    elif op == "sum":
        column = params.get("column")
        if column is not None:
            # Try as-is first, then try converting to int if string
            try:
                result = df[column].sum()
            except KeyError:
                try:
                    result = df[int(column)].sum()
                except (ValueError, KeyError):
                    raise
        else:
            result = df.sum()
    elif op == "mean":
        column = params.get("column")
        if column is not None:
            # Try as-is first, then try converting to int if string
            try:
                result = df[column].mean()
            except KeyError:
                try:
                    result = df[int(column)].mean()
                except (ValueError, KeyError):
                    raise
        else:
            result = df.mean()
    elif op == "groupby":
        # Use canonical parameter names only
        by = params.get("by")
        agg = params.get("aggregation", "count")
        
        if not by:
            raise ValueError("groupby requires 'by' parameter (array of column names to group by)")
        
        # Support both simple and named aggregation
        # Simple: {"amount": "sum"} → column stays "amount"
        # Named: {"total": ["amount", "sum"]} → column renamed to "total" (JSON array)
        # Named: {"total": ("amount", "sum")} → column renamed to "total" (Python tuple for compatibility)
        if isinstance(agg, dict):
            # Check if it's named aggregation (values are tuples/lists) or simple (values are strings)
            first_value = next(iter(agg.values()))
            if isinstance(first_value, (tuple, list)):
                # Named aggregation: {"total": ["amount", "sum"]} or {"total": ("amount", "sum")}
                # Convert lists to tuples for pandas compatibility
                converted_agg = {}
                for key, val in agg.items():
                    if isinstance(val, list):
                        converted_agg[key] = tuple(val)  # Convert JSON array to tuple
                    else:
                        converted_agg[key] = val  # Already a tuple
                result = df.groupby(by).agg(**converted_agg)
            else:
                # Simple aggregation: {"amount": "sum"}
                result = df.groupby(by).agg(agg)
        else:
            # String aggregation: "count", "sum", etc.
            result = df.groupby(by).agg(agg)
        
        # Reset index to make grouped columns accessible as regular columns
        result = result.reset_index()
    elif op == "count":
        result = len(df)
    elif op == "pivot":
        index = params.get("index")
        columns = params.get("columns")
        values = params.get("values")
        
        if not all([index, columns, values]):
            raise ValueError("Pivot requires 'index', 'columns', and 'values' parameters")
        
        result = df.pivot(index=index, columns=columns, values=values)
        result = result.reset_index()  # Reset index to make it a regular column
    elif op == "melt":
        id_vars = params.get("id_vars", [])
        value_vars = params.get("value_vars")
        var_name = params.get("var_name", "variable")
        value_name = params.get("value_name", "value")
        
        result = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name
        )
    elif op == "transpose":
        result = df.transpose()
    elif op == "sort":
        # Sort dataframe by one or more columns
        by = params.get("by")  # Can be string or list
        ascending = params.get("ascending", True)  # Default ascending
        
        if not by:
            raise ValueError("sort requires 'by' parameter (column name or list of column names)")
        
        result = df.sort_values(by=by, ascending=ascending)
    elif op == "head":
        # Take first N rows
        n = params.get("n", 5)  # Default 5 rows
        result = df.head(n)
    elif op == "tail":
        # Take last N rows
        n = params.get("n", 5)  # Default 5 rows
        result = df.tail(n)
    elif op == "idxmax" or op == "idxmin":
        value_column = params.get("value_column")
        label_column = params.get("label_column")
        
        if not value_column:
            raise ValueError(f"{op} requires 'value_column' parameter")
        
        # Find index of max/min value
        if op == "idxmax":
            idx = df[value_column].idxmax()
        else:
            idx = df[value_column].idxmin()
        
        # If label_column specified, return that value from the row
        if label_column:
            result = df.loc[idx, label_column]
        else:
            # Return the entire row as dict
            result = df.loc[idx].to_dict()
    else:
        raise ValueError(f"Unknown operation: {op}")
    
    # Validate result is not None (indicates operation failed)
    if result is None:
        raise ValueError(f"Operation '{op}' failed to produce a result. Check parameters: {params}")
    
    if isinstance(result, pd.DataFrame):
        # Thread-safe registry write
        with _dataframe_lock:
            new_key = f"df_{len(dataframe_registry)}"
            dataframe_registry[new_key] = result
        return {
            "dataframe_key": new_key,
            "result": "DataFrame operation completed",
            "shape": result.shape
        }
    else:
        return {"result": result, "type": type(result).__name__}


def transform_data(dataframe: str, operation: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Transform DataFrame structure (pivot, melt, transpose, reshape)"""
    if params is None:
        params = {}
    
    # Thread-safe registry read
    with _dataframe_lock:
        if dataframe not in dataframe_registry:
            raise ValueError(f"DataFrame {dataframe} not found in registry. Available: {list(dataframe_registry.keys())}")
        df = dataframe_registry[dataframe].copy()
    
    result = None
    
    if operation == "pivot":
        index = params.get("index")
        columns = params.get("columns")
        values = params.get("values")
        
        if not all([index, columns, values]):
            raise ValueError("Pivot requires 'index', 'columns', and 'values' parameters")
        
        result = df.pivot(index=index, columns=columns, values=values)
        result = result.reset_index()  # Reset index to make it a regular column
        
    elif operation == "melt":
        id_vars = params.get("id_vars", [])
        value_vars = params.get("value_vars")
        var_name = params.get("var_name", "variable")
        value_name = params.get("value_name", "value")
        
        result = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name
        )
        
    elif operation == "transpose":
        result = df.transpose()
        
    elif operation == "reshape":
        # Generic reshape - let pandas handle the details
        shape = params.get("shape")
        if shape:
            result = df.values.reshape(shape)
            result = pd.DataFrame(result)
        else:
            raise ValueError("Reshape requires 'shape' parameter")
    else:
        raise ValueError(f"Unknown transform operation: {operation}")
    
    # Thread-safe registry write
    with _dataframe_lock:
        new_key = f"df_{len(dataframe_registry)}"
        dataframe_registry[new_key] = result
    
    logger.info(f"[TRANSFORM] {operation} completed: {dataframe} → {new_key}, shape: {result.shape}")
    
    return {
        "dataframe_key": new_key,
        "operation": operation,
        "shape": result.shape,
        "columns": list(result.columns),
        "sample": result.head(3).to_dict()
    }


async def analyze_image(image_path: str, task: str) -> Dict[str, Any]:
    """Analyze image using vision AI (OCR, description, etc.)"""
    try:
        # Read image and encode to base64
        with open(image_path, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Determine task-specific prompt
        if task == "ocr":
            system_prompt = "You are an OCR tool. Extract and return ONLY the text/numbers you see in the image, nothing else."
            user_prompt = "Extract all text and numbers from this image."
        elif task == "describe":
            system_prompt = "Describe the image content concisely."
            user_prompt = "What do you see in this image?"
        elif task == "detect_objects":
            system_prompt = "List the objects visible in the image."
            user_prompt = "What objects can you identify?"
        elif task == "classify":
            system_prompt = "Classify the image into appropriate categories."
            user_prompt = "What category does this image belong to?"
        else:
            system_prompt = "Analyze this image."
            user_prompt = "Analyze this image."
        
        # Call vision API
        API_KEY = os.getenv("API_KEY")
        OPEN_AI_BASE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        
        # Combine system prompt into user message for vision APIs (some don't support system messages with vision)
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        json_data = {
            "model": "openai/gpt-4o-mini",  # Supports vision
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": combined_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0,
            "max_tokens": 500,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(OPEN_AI_BASE_URL, headers=headers, json=json_data, timeout=120)
            if response.status_code != 200:
                logger.error(f"[VISION_ERROR] Status: {response.status_code}, Response: {response.text}")
            response.raise_for_status()
            data = response.json()
            result_text = data["choices"][0]["message"]["content"].strip()
        
        logger.info(f"[VISION] Task: {task}, Result: {result_text[:200]}")
        return {"result": result_text}
    
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
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
                plt.plot(df[x], df[y])
            elif plot_type == "bar":
                plt.bar(df[x], df[y])
            elif plot_type == "scatter":
                plt.scatter(df[x], df[y])
        else:
            x = spec.get("x", [])
            y = spec.get("y", [])
            plt.plot(x, y)
        
        plt.title(spec.get("title", ""))
        plt.xlabel(spec.get("xlabel", ""))
        plt.ylabel(spec.get("ylabel", ""))
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return {"image_data": f"data:image/png;base64,{img_base64}"}
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        raise


def zip_base64(paths: List[str]) -> Dict[str, Any]:
    """Zip files and return base64 data URI"""
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            for path in paths:
                zf.write(path, os.path.basename(path))
        buf.seek(0)
        zip_base64 = base64.b64encode(buf.read()).decode()
        return {"zip_data": f"data:application/zip;base64,{zip_base64}"}
    except Exception as e:
        logger.error(f"Error creating zip: {e}")
        raise


async def answer_submit(url: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Submit answer to quiz endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=body,
                timeout=30.0,
                headers={"Content-Type": "application/json"}
            )
            
            result = {
                "status_code": response.status_code,
                "success": 200 <= response.status_code < 300
            }
            
            try:
                result["response"] = response.json()
            except:
                result["response"] = response.text
            
            return result
    except Exception as e:
        logger.error(f"Error submitting answer to {url}: {e}")
        raise
