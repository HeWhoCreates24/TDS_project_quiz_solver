"""
Tool definitions for OpenAI function calling
Defines the schema/interface for all available tools that the LLM can use
"""
from typing import Any, Dict, List


def get_tool_usage_examples() -> str:
    """
    Generate tool usage examples dynamically from tool definitions.
    This ensures prompts stay in sync with available tools.
    """
    return """
AVAILABLE TOOLS AND USAGE:

Web & Data Fetching:
- render_js_page(url): Render webpage with JavaScript, extract text/links/code
- fetch_text(url): Fetch raw text content from URL
- fetch_from_api(url, method, headers, body): Call REST APIs with custom headers
- download_file(url): Download binary files (images, audio, PDFs, etc.)

File Parsing:
- parse_csv(path OR url): Parse CSV → produces {"dataframe_key": "df_0"}
- parse_excel(path): Parse Excel files
- parse_json_file(path): Parse JSON files
- parse_html_tables(path_or_html): Extract HTML tables
- parse_pdf_tables(path, pages): Extract tables from PDFs

Text Processing:
- clean_text(text, remove_special_chars, normalize_whitespace): Clean/normalize text
- extract_patterns(text, pattern_type, custom_pattern): Extract emails, URLs, numbers, etc.

Data Transformation:
- dataframe_ops(op, params): All DataFrame operations including transformations
  * CRITICAL STRUCTURE: {"op": "pivot", "params": {"dataframe_key": "df_0", "index": "...", "columns": "...", "values": "..."}}
  * NOT this: {"operation": "pivot", "dataframe": "df_0", ...} ❌
  * NOT this: {"op": "pivot", "dataframe_key": "df_0", ...} ❌
  * Row operations: filter, sum, mean, count, select, groupby
  * Shape operations: pivot, melt, transpose
  * Pivot example: {"op": "pivot", "params": {"dataframe_key": "df_0", "index": "category", "columns": "month", "values": "sales"}}
  * Sum example: {"op": "sum", "params": {"dataframe_key": "df_1", "column": "sales"}}
  * Filter creates NEW dataframe: df_0 → filter → df_1
- calculate_statistics(dataframe, stats, columns): Calculate sum, mean, median, std, etc.
  * IMPORTANT: Use 'dataframe' parameter to specify which dataframe key
  * Example: {"dataframe": "df_0", "stats": ["sum"], "columns": ["sales"]}
  * Use for STATISTICAL analysis on columns

Machine Learning:
- train_linear_regression(dataframe_key, feature_columns, target_column, predict_x): sklearn regression
- apply_ml_model(dataframe, model_type, kwargs): Apply ML models

Multimedia:
- transcribe_audio(audio_path): Speech-to-text transcription
- analyze_image(image_path, task): Vision AI for OCR, description, object detection, classification
  * task="ocr" extracts text/numbers from images
  * Use for ANY image analysis, NOT call_llm (call_llm cannot process images)
- extract_audio_metadata(path): Get audio duration, sample rate, etc.

Visualization:
- create_chart(dataframe, chart_type, x_col, y_col, title, output_path): Create static charts
  * IMPORTANT: Use 'dataframe' parameter (the dataframe_key string, e.g., "df_0")
  * Returns: {"chart_path": "path/to/chart.png", "unique_categories": N}
  * The unique_categories field contains the COUNT of unique values in x_col
  * Use this when quiz asks "how many categories in the chart"
- create_interactive_chart(dataframe, chart_type, x_col, y_col, title): Plotly interactive charts
- make_plot(spec): Custom plotting with detailed specs

Utilities:
- call_llm(prompt, system_prompt, max_tokens, temperature): Text analysis only (NOT for images)
- zip_base64(paths): Create zip archives
- geospatial_analysis(dataframe, analysis_type, kwargs): Distance, geocoding, spatial joins
- generate_narrative(dataframe, summary_stats): Generate natural language from data

CRITICAL PATTERNS:
- Images → download_file + analyze_image (task="ocr")
- Audio → download_file + transcribe_audio
- CSV analysis → parse_csv + dataframe_ops/calculate_statistics
- Filtering → dataframe_ops creates NEW dataframe with new key
- Pivoting/reshaping → parse_csv + dataframe_ops (op="pivot")
  * Example: {"op": "pivot", "params": {"dataframe_key": "df_0", "index": "category", "columns": "month", "values": "sales"}}
  * After pivot, columns become actual data values (e.g., "January", "February")
- After pivot → use calculate_statistics or dataframe_ops on result

PARAMETER NAMING CONVENTIONS:
- dataframe_ops: Use {"op": "...", "params": {"dataframe_key": "df_X", ...}}
  * CRITICAL: Parameter is "op" NOT "operation"
  * CRITICAL: Dataframe identifier goes in params.dataframe_key NOT top-level "dataframe"
  * Works for ALL operations: filter, sum, mean, count, select, groupby, pivot, melt, transpose
  * Pivot example: {"op": "pivot", "params": {"dataframe_key": "df_0", "index": "category", "columns": "month", "values": "sales"}}
  * Sum example: {"op": "sum", "params": {"dataframe_key": "df_1", "column": "January"}}
- calculate_statistics: Use {"dataframe": "df_X", "stats": [...], "columns": [...]}
  * The dataframe identifier is a top-level parameter called 'dataframe'
"""


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Define available tools for OpenAI function calling"""
    return [
        # Web scraping and rendering
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
                            "description": "HTTP method (default: GET)"
                        },
                        "headers": {
                            "type": "object",
                            "description": "Custom headers as key-value pairs",
                            "additionalProperties": {"type": "string"}
                        },
                        "body": {
                            "type": "object",
                            "description": "Request body for POST/PUT as JSON object",
                            "additionalProperties": True
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        # File parsing tools
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
        # Data cleaning tools
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
        # Multimedia processing
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
                "description": "Analyze image content using vision AI. Performs OCR to extract text/numbers from images, describes image content, detects objects, or classifies images. Use this for any task involving reading or understanding image files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to image file or artifact key containing image path (e.g., 'image_data')"
                        },
                        "task": {
                            "type": "string",
                            "enum": ["describe", "ocr", "detect_objects", "classify"],
                            "description": "Analysis task to perform: 'ocr' for extracting text/numbers, 'describe' for image description, 'detect_objects' for object detection, 'classify' for categorization"
                        }
                    },
                    "required": ["image_path", "task"]
                }
            }
        },
        # DataFrame operations
        {
            "type": "function",
            "function": {
                "name": "dataframe_ops",
                "description": "Perform DataFrame operations: filter rows, calculate aggregations (sum/mean), select columns, group by categories, or reshape data (pivot/melt/transpose). Creates new DataFrames for operations that transform structure.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Operation type",
                            "enum": ["filter", "sum", "mean", "count", "select", "groupby", "pivot", "melt", "transpose"]
                        },
                        "params": {
                            "type": "object",
                            "description": "Operation parameters",
                            "properties": {
                                "dataframe_key": {
                                    "type": "string",
                                    "description": "DataFrame to operate on (e.g., 'df_0')"
                                },
                                "condition": {
                                    "type": "string",
                                    "description": "For filter: COMPLETE condition string with column name, operator, and value (e.g., '96903 >= 1371' or 'temperature > 25'). MUST include the full comparison value."
                                },
                                "column": {
                                    "type": "string",
                                    "description": "For sum/mean: column name to aggregate"
                                },
                                "index": {
                                    "type": "string",
                                    "description": "For pivot: column name to use as row index"
                                },
                                "columns": {
                                    "type": "string",
                                    "description": "For pivot: column name whose values become new column headers"
                                },
                                "values": {
                                    "type": "string",
                                    "description": "For pivot: column name containing the data values"
                                },
                                "id_vars": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "For melt: columns to use as identifier variables"
                                },
                                "value_vars": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "For melt: columns to unpivot"
                                }
                            },
                            "required": ["dataframe_key"]
                        }
                    },
                    "required": ["op", "params"]
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
                "name": "train_linear_regression",
                "description": "Train a sklearn linear regression model and make predictions. Use for machine learning and regression tasks. Returns coefficients, R² score, and optional prediction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataframe_key": {
                            "type": "string",
                            "description": "DataFrame key from registry (e.g., 'df_0', 'df_1')"
                        },
                        "feature_columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of column names to use as features/X (e.g., ['x'] or ['feature1', 'feature2'])"
                        },
                        "target_column": {
                            "type": "string",
                            "description": "Column name to predict/y (e.g., 'y', 'price', 'sales')"
                        },
                        "predict_x": {
                            "type": "object",
                            "description": "Optional: Feature values to predict for. Keys must match feature_columns. Example: {'x': 50.0}",
                            "additionalProperties": {
                                "type": "number"
                            }
                        }
                    },
                    "required": ["dataframe_key", "feature_columns", "target_column"]
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
                            "description": "DataFrame key from registry (e.g., 'df_0')"
                        },
                        "model_type": {
                            "type": "string",
                            "enum": ["linear_regression", "logistic_regression", "kmeans", "decision_tree"],
                            "description": "ML model to apply"
                        },
                        "kwargs": {
                            "type": "object",
                            "description": "Model parameters",
                            "properties": {
                                "target_column": {
                                    "type": "string",
                                    "description": "Target/y column for supervised learning"
                                },
                                "feature_columns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Feature/X columns"
                                },
                                "n_clusters": {
                                    "type": "integer",
                                    "description": "Number of clusters for kmeans"
                                }
                            }
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
                            "description": "DataFrame key with geospatial data (e.g., 'df_0')"
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["distance", "geocode", "spatial_join", "buffer"],
                            "description": "Type of geospatial analysis"
                        },
                        "kwargs": {
                            "type": "object",
                            "description": "Analysis parameters",
                            "properties": {
                                "lat_col": {
                                    "type": "string",
                                    "description": "Latitude column name"
                                },
                                "lon_col": {
                                    "type": "string",
                                    "description": "Longitude column name"
                                },
                                "address_col": {
                                    "type": "string",
                                    "description": "Address column for geocoding"
                                }
                            }
                        }
                    },
                    "required": ["analysis_type"]
                }
            }
        },
        # Visualization
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
                            "description": "DataFrame key from registry (e.g., 'df_0')"
                        },
                        "summary_stats": {
                            "type": "object",
                            "description": "Summary statistics to include",
                            "properties": {
                                "mean": {"type": "number"},
                                "median": {"type": "number"},
                                "std": {"type": "number"},
                                "count": {"type": "integer"}
                            },
                            "additionalProperties": True
                        }
                    },
                    "required": ["dataframe"]
                }
            }
        },
        # LLM and utilities
        {
            "type": "function",
            "function": {
                "name": "call_llm",
                "description": "Call an LLM to analyze data, extract information, or generate content.",
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
                            "description": "Plot specification",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "description": "Plot type (line, bar, scatter, etc.)"
                                },
                                "data": {
                                    "type": "object",
                                    "description": "Data source",
                                    "properties": {
                                        "dataframe_key": {"type": "string"},
                                        "x": {"type": "string", "description": "X column"},
                                        "y": {"type": "string", "description": "Y column"}
                                    }
                                },
                                "title": {"type": "string"},
                                "output_path": {"type": "string"}
                            },
                            "required": ["type", "data"]
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
