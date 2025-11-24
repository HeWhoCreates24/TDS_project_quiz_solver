"""
Tool definitions for OpenAI function calling
Defines the schema/interface for all available tools that the LLM can use
"""
from typing import Any, Dict, List


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
        # Data transformation
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
                            "description": "DataFrame key from registry (e.g., 'df_0')"
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["pivot", "melt", "transpose", "reshape"],
                            "description": "Transformation operation"
                        },
                        "params": {
                            "type": "object",
                            "description": "Operation-specific parameters",
                            "properties": {
                                "index": {
                                    "type": "string",
                                    "description": "For pivot: column to use as index"
                                },
                                "columns": {
                                    "type": "string",
                                    "description": "For pivot: column to use as columns"
                                },
                                "values": {
                                    "type": "string",
                                    "description": "For pivot: column to use as values"
                                }
                            }
                        }
                    },
                    "required": ["dataframe", "operation"]
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
        # DataFrame operations
        {
            "type": "function",
            "function": {
                "name": "dataframe_ops",
                "description": "Perform operations on DataFrames: filter rows, calculate sums/means, select columns, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Operation type",
                            "enum": ["filter", "sum", "mean", "count", "select", "groupby"]
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
