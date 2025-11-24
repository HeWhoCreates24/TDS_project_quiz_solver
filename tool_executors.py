"""
Tool execution functions for quiz solving
Handles web scraping, file parsing, data operations, and artifact generation
"""
import os
import json
import base64
import io
import zipfile
import logging
from typing import Any, Dict, List
from tempfile import NamedTemporaryFile
from playwright.async_api import async_playwright
import httpx
import pandas as pd
import matplotlib.pyplot as plt

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
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return {"content": response.text, "status_code": response.status_code}
    except Exception as e:
        logger.error(f"Error fetching text from {url}: {e}")
        raise


async def download_file(url: str) -> Dict[str, Any]:
    """Download file and return metadata"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=60.0)
            response.raise_for_status()
            
            # Save to temp file
            with NamedTemporaryFile(delete=False, suffix=os.path.splitext(url)[1]) as tmp:
                tmp.write(response.content)
                path = tmp.name
            
            return {
                "path": path,
                "size": len(response.content),
                "content_type": response.headers.get("content-type")
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
            raise ValueError("Either path or url must be provided")
        
        # Try reading first to detect if it has headers
        # Read a sample to check if first row looks like data or headers
        sample_df = pd.read_csv(source, nrows=5)
        
        # Heuristic: If the first row (column names) are all numeric, likely no header
        has_header = True
        try:
            pd.to_numeric(sample_df.columns)
            has_header = False
        except:
            has_header = True
        
        # Read the full CSV with appropriate header setting
        if has_header:
            df = pd.read_csv(source)
        else:
            df = pd.read_csv(source, header=None)
        
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
            dataframe_registry[f"df_{len(dataframe_registry)}"] = table
            result[f"table_{i}"] = {
                "dataframe_key": f"df_{len(dataframe_registry)-1}",
                "shape": table.shape
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
        if condition:
            try:
                # Parse condition: "column op value" or "value op column"
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
                    raise ValueError(f"Invalid condition format: {condition}")
            except Exception as e:
                logger.error(f"Error evaluating condition '{condition}': {e}")
                raise ValueError(f"Invalid filter condition: {condition}")
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
