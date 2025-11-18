"""
Modular tools for quiz solving - organized by functionality
"""
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
import httpx
from playwright.async_api import async_playwright
import asyncio

logger = logging.getLogger(__name__)

# ============================================================================
# SCRAPING TOOLS - Website and API data retrieval
# ============================================================================

class ScrapingTools:
    """Tools for scraping websites and APIs"""
    
    @staticmethod
    async def scrape_with_javascript(url: str, wait_for_selector: Optional[str] = None, timeout: int = 30000) -> str:
        """
        Scrape website with JavaScript rendering using Playwright
        
        Args:
            url: Website URL to scrape
            wait_for_selector: CSS selector to wait for before returning
            timeout: Timeout in milliseconds
            
        Returns:
            HTML content as string
        """
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                
                if wait_for_selector:
                    await page.wait_for_selector(wait_for_selector, timeout=timeout)
                
                content = await page.content()
                await browser.close()
                return content
        except Exception as e:
            logger.error(f"[SCRAPE_JS] Failed to scrape {url}: {e}")
            raise
    
    @staticmethod
    async def fetch_from_api(url: str, method: str = "GET", headers: Optional[Dict] = None, 
                            body: Optional[Dict] = None, timeout: int = 30) -> Dict[str, Any]:
        """
        Fetch data from API with optional headers
        
        Args:
            url: API endpoint URL
            method: HTTP method (GET, POST, PUT, DELETE)
            headers: Optional headers dict
            body: Optional request body
            timeout: Request timeout in seconds
            
        Returns:
            Response data as dict
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=headers, json=body)
                elif method.upper() == "PUT":
                    response = await client.put(url, headers=headers, json=body)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return {"status_code": response.status_code, "data": response.json()}
        except Exception as e:
            logger.error(f"[API_FETCH] Failed to fetch from {url}: {e}")
            raise
    
    @staticmethod
    async def extract_html_text(html: str, selector: Optional[str] = None) -> str:
        """
        Extract text content from HTML
        
        Args:
            html: HTML content
            selector: Optional CSS selector to extract specific elements
            
        Returns:
            Extracted text
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            if selector:
                elements = soup.select(selector)
                text = '\n'.join([elem.get_text(strip=True) for elem in elements])
            else:
                text = soup.get_text(strip=True)
            
            return text
        except Exception as e:
            logger.error(f"[HTML_EXTRACT] Failed to extract text: {e}")
            raise


# ============================================================================
# DATA CLEANSING TOOLS - Text, data, PDF processing
# ============================================================================

class CleansingTools:
    """Tools for data cleansing and preprocessing"""
    
    @staticmethod
    def clean_text(text: str, lowercase: bool = False, remove_special: bool = True, 
                   remove_whitespace: bool = True) -> str:
        """
        Clean text content
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_special: Remove special characters
            remove_whitespace: Remove extra whitespace
            
        Returns:
            Cleaned text
        """
        try:
            if lowercase:
                text = text.lower()
            
            if remove_special:
                text = re.sub(r'[^a-zA-Z0-9\s\.\,\-\:]', '', text)
            
            if remove_whitespace:
                text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            logger.error(f"[TEXT_CLEAN] Failed to clean text: {e}")
            raise
    
    @staticmethod
    def extract_from_pdf(pdf_path: str, pages: Optional[List[int]] = None) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            pages: Optional list of page numbers to extract (0-indexed)
            
        Returns:
            Extracted text
        """
        try:
            import PyPDF2
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page_list = pages if pages else range(len(reader.pages))
                for page_num in page_list:
                    page = reader.pages[page_num]
                    text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"[PDF_EXTRACT] Failed to extract from PDF: {e}")
            raise
    
    @staticmethod
    def parse_csv_data(csv_content: str, delimiter: str = ',') -> pd.DataFrame:
        """
        Parse CSV content into DataFrame
        
        Args:
            csv_content: CSV content as string
            delimiter: CSV delimiter (default: comma)
            
        Returns:
            Pandas DataFrame
        """
        try:
            from io import StringIO
            return pd.read_csv(StringIO(csv_content), delimiter=delimiter)
        except Exception as e:
            logger.error(f"[CSV_PARSE] Failed to parse CSV: {e}")
            raise
    
    @staticmethod
    def parse_json_data(json_content: str) -> Union[Dict, List]:
        """
        Parse JSON content
        
        Args:
            json_content: JSON content as string
            
        Returns:
            Parsed JSON object
        """
        try:
            return json.loads(json_content)
        except Exception as e:
            logger.error(f"[JSON_PARSE] Failed to parse JSON: {e}")
            raise
    
    @staticmethod
    def extract_structured_data(text: str, pattern: str) -> List[str]:
        """
        Extract data using regex pattern
        
        Args:
            text: Input text
            pattern: Regex pattern
            
        Returns:
            List of matches
        """
        try:
            return re.findall(pattern, text)
        except Exception as e:
            logger.error(f"[PATTERN_EXTRACT] Failed to extract with pattern: {e}")
            raise


# ============================================================================
# DATA PROCESSING TOOLS - Transformation, aggregation, transcription
# ============================================================================

class ProcessingTools:
    """Tools for data processing and transformation"""
    
    @staticmethod
    def transform_dataframe(df: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply multiple transformation operations to DataFrame
        
        Args:
            df: Input DataFrame
            operations: List of operations, each with 'type' and operation-specific params
                       Examples: 
                       - {"type": "select_columns", "columns": ["col1", "col2"]}
                       - {"type": "rename", "mapping": {"old_name": "new_name"}}
                       - {"type": "filter", "column": "col", "condition": ">", "value": 10}
                       - {"type": "sort", "by": "col1", "ascending": True}
            
        Returns:
            Transformed DataFrame
        """
        try:
            result = df.copy()
            for op in operations:
                op_type = op.get("type")
                
                if op_type == "select_columns":
                    result = result[op["columns"]]
                elif op_type == "rename":
                    result = result.rename(columns=op["mapping"])
                elif op_type == "filter":
                    col = op["column"]
                    condition = op["condition"]
                    value = op["value"]
                    if condition == ">":
                        result = result[result[col] > value]
                    elif condition == "<":
                        result = result[result[col] < value]
                    elif condition == "==":
                        result = result[result[col] == value]
                    elif condition == "!=":
                        result = result[result[col] != value]
                elif op_type == "sort":
                    result = result.sort_values(by=op["by"], ascending=op.get("ascending", True))
                elif op_type == "group_by":
                    agg_funcs = op.get("aggregations", {})
                    result = result.groupby(op["by"]).agg(agg_funcs).reset_index()
                elif op_type == "fill_na":
                    result = result.fillna(op.get("value", 0))
                elif op_type == "drop_duplicates":
                    result = result.drop_duplicates(subset=op.get("subset"))
            
            return result
        except Exception as e:
            logger.error(f"[TRANSFORM] Failed to transform DataFrame: {e}")
            raise
    
    @staticmethod
    def aggregate_data(df: pd.DataFrame, group_by: List[str], aggregations: Dict[str, str]) -> pd.DataFrame:
        """
        Aggregate data with groupby
        
        Args:
            df: Input DataFrame
            group_by: Columns to group by
            aggregations: Dict mapping column names to aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        try:
            return df.groupby(group_by).agg(aggregations).reset_index()
        except Exception as e:
            logger.error(f"[AGGREGATE] Failed to aggregate data: {e}")
            raise
    
    @staticmethod
    def reshape_data(df: pd.DataFrame, reshape_type: str, **kwargs) -> pd.DataFrame:
        """
        Reshape DataFrame (pivot, melt, etc.)
        
        Args:
            df: Input DataFrame
            reshape_type: Type of reshape ('pivot', 'melt', 'transpose')
            **kwargs: Operation-specific parameters
            
        Returns:
            Reshaped DataFrame
        """
        try:
            if reshape_type == "pivot":
                return df.pivot_table(
                    index=kwargs.get("index"),
                    columns=kwargs.get("columns"),
                    values=kwargs.get("values"),
                    aggfunc=kwargs.get("aggfunc", "first")
                )
            elif reshape_type == "melt":
                return df.melt(
                    id_vars=kwargs.get("id_vars"),
                    value_vars=kwargs.get("value_vars"),
                    var_name=kwargs.get("var_name", "variable"),
                    value_name=kwargs.get("value_name", "value")
                )
            elif reshape_type == "transpose":
                return df.T
            else:
                raise ValueError(f"Unknown reshape type: {reshape_type}")
        except Exception as e:
            logger.error(f"[RESHAPE] Failed to reshape data: {e}")
            raise
    
    @staticmethod
    def transcribe_content(content: str, format_type: str = "text") -> str:
        """
        Transcribe or convert content format
        
        Args:
            content: Input content
            format_type: Format type (text, markdown, html, etc.)
            
        Returns:
            Transcribed content
        """
        try:
            if format_type == "markdown":
                return content  # Could add markdown parsing here
            elif format_type == "html":
                from html import unescape
                return unescape(content)
            else:
                return content
        except Exception as e:
            logger.error(f"[TRANSCRIBE] Failed to transcribe content: {e}")
            raise


# ============================================================================
# ANALYSIS TOOLS - Filtering, sorting, aggregation, statistical/ML models
# ============================================================================

class AnalysisTools:
    """Tools for data analysis and modeling"""
    
    @staticmethod
    def filter_data(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply multiple filters to DataFrame
        
        Args:
            df: Input DataFrame
            filters: List of filter conditions
                    [{"column": "age", "operator": ">", "value": 25}, ...]
            
        Returns:
            Filtered DataFrame
        """
        try:
            result = df.copy()
            for f in filters:
                col = f["column"]
                op = f["operator"]
                val = f["value"]
                
                if op == ">":
                    result = result[result[col] > val]
                elif op == "<":
                    result = result[result[col] < val]
                elif op == ">=":
                    result = result[result[col] >= val]
                elif op == "<=":
                    result = result[result[col] <= val]
                elif op == "==":
                    result = result[result[col] == val]
                elif op == "!=":
                    result = result[result[col] != val]
                elif op == "in":
                    result = result[result[col].isin(val)]
                elif op == "contains":
                    result = result[result[col].str.contains(val, na=False)]
            
            return result
        except Exception as e:
            logger.error(f"[FILTER] Failed to filter data: {e}")
            raise
    
    @staticmethod
    def sort_data(df: pd.DataFrame, sort_by: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Sort DataFrame by multiple columns
        
        Args:
            df: Input DataFrame
            sort_by: List of sort specs [{"column": "col1", "ascending": True}, ...]
            
        Returns:
            Sorted DataFrame
        """
        try:
            columns = [s["column"] for s in sort_by]
            ascending = [s.get("ascending", True) for s in sort_by]
            return df.sort_values(by=columns, ascending=ascending)
        except Exception as e:
            logger.error(f"[SORT] Failed to sort data: {e}")
            raise
    
    @staticmethod
    def calculate_statistics(df: pd.DataFrame, columns: List[str], stats: List[str]) -> Dict[str, Any]:
        """
        Calculate statistical measures
        
        Args:
            df: Input DataFrame
            columns: Columns to analyze
            stats: List of statistics ('mean', 'median', 'std', 'min', 'max', 'count', etc.)
            
        Returns:
            Dictionary of statistics
        """
        try:
            results = {}
            for col in columns:
                col_stats = {}
                for stat in stats:
                    if stat == "mean":
                        col_stats["mean"] = df[col].mean()
                    elif stat == "median":
                        col_stats["median"] = df[col].median()
                    elif stat == "std":
                        col_stats["std"] = df[col].std()
                    elif stat == "min":
                        col_stats["min"] = df[col].min()
                    elif stat == "max":
                        col_stats["max"] = df[col].max()
                    elif stat == "count":
                        col_stats["count"] = df[col].count()
                    elif stat == "sum":
                        col_stats["sum"] = df[col].sum()
                    elif stat == "variance":
                        col_stats["variance"] = df[col].var()
                results[col] = col_stats
            return results
        except Exception as e:
            logger.error(f"[STATISTICS] Failed to calculate statistics: {e}")
            raise
    
    @staticmethod
    def apply_ml_model(df: pd.DataFrame, model_type: str, **kwargs) -> Dict[str, Any]:
        """
        Apply machine learning model
        
        Args:
            df: Input DataFrame
            model_type: Type of model ('linear_regression', 'kmeans', 'random_forest', etc.)
            **kwargs: Model-specific parameters
            
        Returns:
            Model results
        """
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.cluster import KMeans
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            if model_type == "linear_regression":
                X = df[kwargs["features"]]
                y = df[kwargs["target"]]
                model = LinearRegression()
                model.fit(X, y)
                return {
                    "model_type": "linear_regression",
                    "coefficients": dict(zip(kwargs["features"], model.coef_)),
                    "intercept": model.intercept_,
                    "score": model.score(X, y)
                }
            elif model_type == "kmeans":
                X = df[kwargs["features"]]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model = KMeans(n_clusters=kwargs.get("n_clusters", 3))
                clusters = model.fit_predict(X_scaled)
                return {
                    "model_type": "kmeans",
                    "n_clusters": kwargs.get("n_clusters", 3),
                    "inertia": model.inertia_,
                    "clusters": clusters.tolist()
                }
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            logger.error(f"[ML_MODEL] Failed to apply model: {e}")
            raise
    
    @staticmethod
    def geospatial_analysis(df: pd.DataFrame, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """
        Perform geospatial analysis
        
        Args:
            df: Input DataFrame with location data
            analysis_type: Type of analysis ('distance', 'clustering', 'bounds', etc.)
            **kwargs: Analysis-specific parameters
            
        Returns:
            Analysis results
        """
        try:
            from math import radians, sin, cos, sqrt, atan2
            
            if analysis_type == "distance":
                lat1, lon1 = kwargs["point1"]
                lat2, lon2 = kwargs["point2"]
                R = 6371  # Earth radius in km
                dlat = radians(lat2 - lat1)
                dlon = radians(lon2 - lon1)
                a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                return {"distance_km": R * c}
            else:
                raise ValueError(f"Unsupported geospatial analysis: {analysis_type}")
        except Exception as e:
            logger.error(f"[GEOSPATIAL] Failed to perform geospatial analysis: {e}")
            raise


# ============================================================================
# VISUALIZATION TOOLS - Charts, graphs, narratives, slides
# ============================================================================

class VisualizationTools:
    """Tools for data visualization"""
    
    @staticmethod
    def create_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str, 
                    title: str = "", output_path: Optional[str] = None) -> str:
        """
        Create a chart from DataFrame
        
        Args:
            df: Input DataFrame
            chart_type: Type of chart ('line', 'bar', 'scatter', 'histogram', 'pie', 'box')
            x_col: Column for X-axis
            y_col: Column for Y-axis (or column for pie/histogram)
            title: Chart title
            output_path: Optional path to save chart as image
            
        Returns:
            Path to saved chart or chart data
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == "line":
                ax.plot(df[x_col], df[y_col], marker='o')
            elif chart_type == "bar":
                ax.bar(df[x_col], df[y_col])
            elif chart_type == "scatter":
                ax.scatter(df[x_col], df[y_col])
            elif chart_type == "histogram":
                ax.hist(df[y_col], bins=20)
            elif chart_type == "pie":
                ax.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%')
            elif chart_type == "box":
                ax.boxplot([df[col] for col in df.columns])
            
            ax.set_title(title)
            ax.set_xlabel(x_col)
            if chart_type != "pie":
                ax.set_ylabel(y_col)
            
            if output_path:
                fig.savefig(output_path, dpi=100, bbox_inches='tight')
                logger.info(f"[CHART_SAVED] Chart saved to {output_path}")
                return output_path
            else:
                return "chart_created"
        except Exception as e:
            logger.error(f"[CHART] Failed to create chart: {e}")
            raise
    
    @staticmethod
    def create_interactive_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str,
                                title: str = "", output_path: Optional[str] = None) -> str:
        """
        Create interactive chart using Plotly
        
        Args:
            df: Input DataFrame
            chart_type: Type of chart ('line', 'bar', 'scatter', etc.)
            x_col: Column for X-axis
            y_col: Column for Y-axis
            title: Chart title
            output_path: Optional path to save as HTML
            
        Returns:
            Path to saved interactive chart or chart data
        """
        try:
            import plotly.express as px
            
            if chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, title=title)
            elif chart_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, title=title)
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title=title)
            elif chart_type == "histogram":
                fig = px.histogram(df, x=y_col, title=title)
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            if output_path:
                fig.write_html(output_path)
                logger.info(f"[INTERACTIVE_CHART] Saved to {output_path}")
                return output_path
            else:
                return "interactive_chart_created"
        except Exception as e:
            logger.error(f"[INTERACTIVE_CHART] Failed to create chart: {e}")
            raise
    
    @staticmethod
    def generate_narrative(df: pd.DataFrame, summary_stats: Dict[str, Any]) -> str:
        """
        Generate text narrative from data
        
        Args:
            df: Input DataFrame
            summary_stats: Dictionary of summary statistics
            
        Returns:
            Narrative text
        """
        try:
            narrative = f"Dataset Overview:\n"
            narrative += f"- Total Records: {len(df)}\n"
            narrative += f"- Columns: {', '.join(df.columns)}\n"
            narrative += f"- Data Types: {dict(df.dtypes)}\n\n"
            
            for col, stats in summary_stats.items():
                narrative += f"{col}:\n"
                for stat_name, stat_value in stats.items():
                    narrative += f"  - {stat_name}: {stat_value:.2f}\n"
            
            return narrative
        except Exception as e:
            logger.error(f"[NARRATIVE] Failed to generate narrative: {e}")
            raise
    
    @staticmethod
    def create_presentation_slide(title: str, content: Dict[str, Any], 
                                 output_path: Optional[str] = None) -> str:
        """
        Create presentation slide
        
        Args:
            title: Slide title
            content: Slide content (text, images, charts)
            output_path: Optional path to save presentation
            
        Returns:
            Path to presentation or slide data
        """
        try:
            slide_data = {
                "title": title,
                "timestamp": datetime.now().isoformat(),
                "content": content
            }
            
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(slide_data, f, indent=2)
                logger.info(f"[SLIDE_CREATED] Slide saved to {output_path}")
                return output_path
            else:
                return json.dumps(slide_data)
        except Exception as e:
            logger.error(f"[SLIDE] Failed to create slide: {e}")
            raise


# ============================================================================
# MULTIMEDIA TOOLS - Audio, Video, Image processing
# ============================================================================

class MultimediaTools:
    """Tools for processing multimedia content"""
    
    @staticmethod
    async def download_file(url: str, output_path: Optional[str] = None) -> str:
        """Download file from URL"""
        import os
        import tempfile
        from urllib.parse import urlparse
        
        try:
            if not output_path:
                parsed = urlparse(url)
                ext = os.path.splitext(parsed.path)[1] or '.dat'
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, f"download_{abs(hash(url))}{ext}")
            
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
            
            logger.info(f"[DOWNLOAD] Downloaded {url} to {output_path} ({len(response.content)} bytes)")
            return output_path
        except Exception as e:
            logger.error(f"[DOWNLOAD] Failed: {e}")
            raise
    
    @staticmethod
    def extract_audio_metadata(audio_path: str) -> Dict[str, Any]:
        """Extract metadata from audio file"""
        try:
            import wave
            import os
            
            metadata = {
                "path": audio_path,
                "size_bytes": os.path.getsize(audio_path),
                "exists": os.path.exists(audio_path),
                "filename": os.path.basename(audio_path)
            }
            
            try:
                with wave.open(audio_path, 'rb') as audio:
                    metadata.update({
                        "channels": audio.getnchannels(),
                        "sample_width": audio.getsampwidth(),
                        "framerate": audio.getframerate(),
                        "frames": audio.getnframes(),
                        "duration_seconds": audio.getnframes() / audio.getframerate()
                    })
            except:
                pass
            
            return metadata
        except Exception as e:
            logger.error(f"[AUDIO_META] Failed: {e}")
            return {"error": str(e), "path": audio_path}
    
    @staticmethod
    async def transcribe_audio(audio_path: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio file to text using OpenAI Whisper API"""
        try:
            import os
            
            if not api_key:
                api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                return {"error": "No API key provided for transcription", "text": ""}
            
            # Use OpenRouter's Whisper endpoint or OpenAI directly
            async with httpx.AsyncClient(timeout=120.0) as client:
                with open(audio_path, 'rb') as audio_file:
                    files = {'file': audio_file}
                    data = {'model': 'openai/whisper-1'}
                    headers = {'Authorization': f'Bearer {api_key}'}
                    
                    # Try OpenAI Whisper API
                    response = await client.post(
                        'https://aipipe.org/openrouter/v1/audio/transcriptions',
                        headers=headers,
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"[TRANSCRIBE] Success: {result.get('text', '')[:100]}...")
                        return {
                            "text": result.get('text', ''),
                            "success": True,
                            "audio_path": audio_path
                        }
                    else:
                        logger.error(f"[TRANSCRIBE] Failed: {response.status_code} {response.text}")
                        return {
                            "error": f"API error: {response.status_code}",
                            "text": "",
                            "success": False
                        }
        except Exception as e:
            logger.error(f"[TRANSCRIBE] Failed: {e}")
            return {"error": str(e), "text": "", "success": False}


# ============================================================================
# Unified Tool Registry
# ============================================================================

class ToolRegistry:
    """Registry of all available tools organized by category"""
    
    TOOLS = {
        "scraping": {
            "scrape_with_javascript": ScrapingTools.scrape_with_javascript,
            "fetch_from_api": ScrapingTools.fetch_from_api,
            "extract_html_text": ScrapingTools.extract_html_text,
        },
        "cleansing": {
            "clean_text": CleansingTools.clean_text,
            "extract_from_pdf": CleansingTools.extract_from_pdf,
            "parse_csv_data": CleansingTools.parse_csv_data,
            "parse_json_data": CleansingTools.parse_json_data,
            "extract_structured_data": CleansingTools.extract_structured_data,
        },
        "processing": {
            "transform_dataframe": ProcessingTools.transform_dataframe,
            "aggregate_data": ProcessingTools.aggregate_data,
            "reshape_data": ProcessingTools.reshape_data,
            "transcribe_content": ProcessingTools.transcribe_content,
        },
        "analysis": {
            "filter_data": AnalysisTools.filter_data,
            "sort_data": AnalysisTools.sort_data,
            "calculate_statistics": AnalysisTools.calculate_statistics,
            "apply_ml_model": AnalysisTools.apply_ml_model,
            "geospatial_analysis": AnalysisTools.geospatial_analysis,
        },
        "visualization": {
            "create_chart": VisualizationTools.create_chart,
            "create_interactive_chart": VisualizationTools.create_interactive_chart,
            "generate_narrative": VisualizationTools.generate_narrative,
            "create_presentation_slide": VisualizationTools.create_presentation_slide,
        },
        "multimedia": {
            "download_file": MultimediaTools.download_file,
            "extract_audio_metadata": MultimediaTools.extract_audio_metadata,
            "transcribe_audio": MultimediaTools.transcribe_audio,
        },
    }
    
    @classmethod
    def get_tool(cls, category: str, tool_name: str):
        """Get a tool by category and name"""
        if category not in cls.TOOLS:
            raise ValueError(f"Unknown tool category: {category}")
        if tool_name not in cls.TOOLS[category]:
            raise ValueError(f"Unknown tool in {category}: {tool_name}")
        return cls.TOOLS[category][tool_name]
    
    @classmethod
    def list_tools(cls, category: Optional[str] = None) -> Dict[str, Any]:
        """List available tools"""
        if category:
            return list(cls.TOOLS.get(category, {}).keys())
        return cls.TOOLS
    
    @classmethod
    def get_tool_info(cls) -> str:
        """Get formatted info about all available tools"""
        info = "Available Tools:\n\n"
        for category, tools in cls.TOOLS.items():
            info += f"{category.upper()}:\n"
            for tool_name in tools.keys():
                info += f"  - {tool_name}\n"
            info += "\n"
        return info
