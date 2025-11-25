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
  * Returns: {"status_code": 200, "data": {...}} - JSON response object
  * NOT a dataframe - use call_llm for simple queries or parse_json_file to convert
- download_file(url): Download binary files (images, audio, PDFs, etc.)

File Parsing:
- parse_csv(path OR url): Parse CSV → produces {"dataframe_key": "df_0"}
- parse_excel(path): Parse Excel files → produces dataframe
- parse_json_file(path): Parse JSON to dataframe → produces {"dataframe_key": "df_X"}
  * Use this to convert JSON to dataframe for dataframe_ops
  * For simple JSON queries, use call_llm instead
- parse_html_tables(path_or_html): Extract HTML tables
- parse_pdf_tables(path, pages): Extract tables from PDFs

Text Processing:
- clean_text(text, remove_special_chars, normalize_whitespace): Clean/normalize text
- extract_patterns(text, pattern_type, custom_pattern): Extract patterns from text
  * pattern_type: "email", "url", "phone", "date", "number" (SINGULAR, not "emails")
  * Returns: {"matches": [...], "count": N}
  * Example: {"text": "Contact support@example.com", "pattern_type": "email"}

Data Transformation:
- dataframe_ops(op, params): All DataFrame operations including transformations
  * ONLY works on DATAFRAMES (from parse_csv, parse_excel, parse_json_file)
  * Does NOT work on JSON from fetch_from_api - that's just a dict
  * Dataframe reference: Use params.dataframe_key = "{{artifact_name}}" with template markers
  * CRITICAL STRUCTURE: {"op": "pivot", "params": {"dataframe_key": "{{df_0}}", "index": "...", "columns": "...", "values": "..."}}
  * NOT this: {"operation": "pivot", "dataframe": "df_0", ...} ❌
  * NOT this: {"op": "pivot", "dataframe_key": "df_0", ...} ❌ (missing template markers)
  * Row operations: filter, sum, mean, count, select, groupby
  * Shape operations: pivot, melt, transpose
  * Pivot example: {"op": "pivot", "params": {"dataframe_key": "{{df_0}}", "index": "category", "columns": "month", "values": "sales"}}
  * Sum example: {"op": "sum", "params": {"dataframe_key": "{{df_1}}", "column": "sales"}}
  * Filter creates NEW dataframe: df_0 → filter → df_1

- calculate_statistics(dataframe, stats, columns): Calculate sum, mean, median, std, etc.
  * ONLY works on DATAFRAMES (dataframe_key like "df_0")
  * Dataframe reference: Use dataframe = "{{artifact_name}}" with template markers
  * IMPORTANT: Use top-level 'dataframe' parameter to specify which dataframe
  * Example: {"dataframe": "{{df_0}}", "stats": ["sum"], "columns": ["sales"]}
  * Use for STATISTICAL analysis on columns

Machine Learning:
- train_linear_regression(dataframe_key, feature_columns, target_column, predict_x): sklearn regression
  * Dataframe reference: Use dataframe_key = "{{artifact_name}}" with template markers
  * IMPORTANT: Use top-level 'dataframe_key' parameter to specify which dataframe
  * Example: {"dataframe_key": "{{df_0}}", "feature_columns": ["x"], "target_column": "y", "predict_x": {"x": 50}}
  * Returns model coefficients and optional prediction

- apply_ml_model(dataframe, model_type, kwargs): Apply ML models

Multimedia:
- transcribe_audio(audio_path): Speech-to-text transcription
- analyze_image(image_path, task): Vision AI for OCR, description, object detection, classification
  * task="ocr" extracts text/numbers from images
  * Use for ANY image analysis, NOT call_llm (call_llm cannot process images)
- extract_audio_metadata(path): Get audio duration, sample rate, etc.

Visualization:
- create_chart(dataframe, chart_type, x_col, y_col, title, output_path): Create static charts
  * Dataframe reference: Use dataframe = "{{artifact_name}}" with template markers
  * IMPORTANT: Use 'dataframe' parameter (the dataframe_key string, e.g., "df_0")
  * output_path is OPTIONAL - omit this parameter when chart saving is not needed
  * Returns: {"chart_path": "path/to/chart.png", "unique_categories": N}
  * The unique_categories field contains the COUNT of unique values in x_col
  * Use this when quiz asks "how many categories in the chart"
  * Example (no saving): {"dataframe": "{{df_0}}", "chart_type": "bar", "x_col": "category", "y_col": "sales"}
  * Example (with saving): {"dataframe": "{{df_0}}", "chart_type": "bar", "x_col": "category", "y_col": "sales", "output_path": "/absolute/path/to/chart.png"}
- create_interactive_chart(dataframe, chart_type, x_col, y_col, title): Plotly interactive charts
- make_plot(spec): Custom plotting with detailed specs

Utilities:
- call_llm(prompt, system_prompt, max_tokens, temperature): Text analysis only (NOT for images)
  * Use for simple JSON queries (find max, filter, etc.) when data is NOT a dataframe
  * Example: Find product with highest price from JSON → use call_llm
  * For dataframes, use dataframe_ops or calculate_statistics instead
  
  **HOW TO WRITE EFFECTIVE call_llm PROMPTS:**
  * Be SPECIFIC about output format: "Return only the product ID as a plain string"
  * NOT generic: "Find the product with highest price and return its id" ❌
  * SPECIFIC: "Extract the 'id' field from the product with the highest 'price' value. Return ONLY the id value (e.g., 'P004'), not JSON, not an object, just the raw string value." ✅
  * System prompt should specify: "Extract the product id. Return ONLY the id value like 'P004', not wrapped in JSON."
  * Bad example: LLM returns {"id": "P004"} when you wanted just "P004"
  * Good example: LLM returns "P004" because prompt was explicit
  * Key principle: Tell the LLM EXACTLY what format you want in the response

- zip_base64(paths): Create zip archives
- geospatial_analysis(dataframe, analysis_type, kwargs): Distance, geocoding, spatial joins
- generate_narrative(dataframe, summary_stats): Generate natural language from data

CRITICAL PATTERNS:
- Images → download_file + analyze_image (task="ocr")
- Audio → download_file + transcribe_audio
- CSV/Excel analysis → parse_csv/parse_excel + dataframe_ops/calculate_statistics
- JSON queries → fetch_from_api + call_llm (for simple queries)
- JSON to dataframe → download_file + parse_json_file + dataframe_ops
- Filtering → dataframe_ops creates NEW dataframe with new key
- Pivoting/reshaping → parse_csv + dataframe_ops (op="pivot")
  * Example: {"op": "pivot", "params": {"dataframe_key": "df_0", "index": "category", "columns": "month", "values": "sales"}}
  * After pivot, columns become actual data values (e.g., "January", "February")
- After pivot → use calculate_statistics or dataframe_ops on result

**DATAFRAME REFERENCING (CRITICAL):**
- Parse tools (parse_csv, parse_excel, parse_json_file) store dataframes in a registry
- They return artifacts with metadata: {"dataframe_key": "df_X", "shape": ..., "columns": ...}
- The "dataframe_key" field shows which registry key the dataframe is stored under
- To use a dataframe in subsequent tools, reference the ARTIFACT NAME with template markers {{}}

**HOW TO REFERENCE DATAFRAMES:**
1. Parse tool produces artifact (e.g., "df_0") with metadata: {"dataframe_key": "df_2", ...}
2. In subsequent tools, use: {"dataframe_key": "{{df_0}}", ...}
3. Executor resolves {{df_0}} → looks up artifact → extracts "df_2" → passes to tool

**EXAMPLES:**
```
# Step 1: Parse CSV creates artifact "df_0" 
parse_csv produces: {"dataframe_key": "df_2", "columns": ["x", "y"], ...}

# Step 2: Use {{df_0}} to reference it
CORRECT: {"dataframe_key": "{{df_0}}", "feature_columns": ["x"], ...}  ✅
WRONG:   {"dataframe_key": "df_0", ...}                                ❌ No template markers
WRONG:   {"dataframe_key": "df_2", ...}                                ❌ Hardcoded registry key
```

**DIFFERENT TOOLS USE DIFFERENT PARAMETER NAMES:**

* **dataframe_ops** - Uses nested params.dataframe_key:
  - Structure: {"op": "filter", "params": {"dataframe_key": "{{df_0}}", "condition": "age > 30"}}
  - Example: {"op": "sum", "params": {"dataframe_key": "{{df_1}}", "column": "sales"}}

* **calculate_statistics** - Uses top-level dataframe parameter:
  - Structure: {"dataframe": "{{df_0}}", "stats": ["mean"], "columns": ["score"]}
  - Example: {"dataframe": "{{df_2}}", "stats": ["sum"], "columns": ["revenue"]}

* **train_linear_regression** - Uses top-level dataframe_key parameter:
  - Structure: {"dataframe_key": "{{df_0}}", "feature_columns": ["x"], "target_column": "y"}
  - Example: {"dataframe_key": "{{regression_data}}", "feature_columns": ["x"], "target_column": "y", "predict_x": {"x": 50}}

**KEY RULES:**
- ALWAYS use template markers {{artifact_name}} for dataframe references
- Use the exact parameter name for each tool (dataframe_key vs dataframe vs params.dataframe_key)
- Reference the artifact name (what you called it), not the registry key (executor resolves this)
- Works for ANY artifact name: {{df_0}}, {{my_data}}, {{regression_data}}, etc.

COMMON WORKFLOWS:

1. API FETCH WITH NESTED FIELD:
   Tool: fetch_from_api (NOT "call_api")
   Pattern: fetch_from_api returns {"status_code": 200, "data": {...}}
   Example task: Extract 'secret_code' from API response
   Solution: Use fetch_from_api, reference result with dot notation
   Key point: Artifact naming must match between produces and final_answer_spec
   
2. VISUALIZATION + COUNT:
   Tool: create_chart (NOT dataframe_ops count)
   Pattern: create_chart returns {"chart_path": "...", "unique_categories": N}
   Example task: "Create bar chart and count how many categories"
   Solution: Use create_chart, reference .unique_categories field
   Key point: The unique_categories field automatically contains the count
   
3. DATA TRANSFORMATION:
   Tool: dataframe_ops with op="pivot"
   Pattern: Pivot restructures data shape (rows → columns)
   Example task: "Pivot by category and month, sum January sales"
   Solution: First pivot with {"op": "pivot", "params": {"dataframe_key": "{{df_0}}", ...}}
   Then calculate_statistics on result: {"dataframe": "{{pivoted}}", "stats": ["sum"], ...}
   Key point: After pivot, month names become column names

4. MACHINE LEARNING WORKFLOW:
   Tool: train_linear_regression
   Pattern: Parse CSV → Train model → Get prediction
   Example: {"dataframe_key": "{{df_0}}", "feature_columns": ["x"], "target_column": "y", "predict_x": {"x": 50}}
   Key point: Use {{artifact_name}} to reference the parsed dataframe

PARAMETER NAMING CONVENTIONS:
- dataframe_ops: Use {"op": "...", "params": {"dataframe_key": "{{artifact}}", ...}}
  * CRITICAL: Parameter is "op" NOT "operation"
  * CRITICAL: Dataframe identifier goes in params.dataframe_key with template markers
  * Works for ALL operations: filter, sum, mean, count, select, groupby, pivot, melt, transpose
  * Pivot example: {"op": "pivot", "params": {"dataframe_key": "{{df_0}}", "index": "category", "columns": "month", "values": "sales"}}
  * Sum example: {"op": "sum", "params": {"dataframe_key": "{{df_1}}", "column": "January"}}
  
- calculate_statistics: Use {"dataframe": "{{artifact}}", "stats": [...], "columns": [...]}
  * The dataframe identifier is a top-level parameter called 'dataframe' with template markers
  * Example: {"dataframe": "{{df_0}}", "stats": ["mean", "sum"], "columns": ["revenue"]}

- train_linear_regression: Use {"dataframe_key": "{{artifact}}", "feature_columns": [...], "target_column": "..."}
  * The dataframe identifier is a top-level parameter called 'dataframe_key' with template markers
  * Example: {"dataframe_key": "{{regression_data}}", "feature_columns": ["x"], "target_column": "y"}

ARTIFACT REFERENCE RULES:
- parse_csv produces artifact with: {"dataframe_key": "df_X", ...} where X is auto-incremented
- To use this dataframe: Reference the ARTIFACT NAME with template markers {{artifact_name}}
- Executor resolves {{artifact_name}} → extracts dataframe_key → passes to tool
- Example: parse_csv creates "df_0" artifact → use {"dataframe_key": "{{df_0}}"} in next tool
- dataframe_ops filter creates NEW artifact: df_0 → filter → df_1 (new artifact with new registry key)
- parse_csv REQUIRES URL OR FILE PATH - NEVER use {{artifact}} syntax in parse_csv!
  * ✅ CORRECT: {"tool_name": "parse_csv", "inputs": {"url": "http://example.com/data.csv"}}
  * ❌ WRONG: {"tool_name": "parse_csv", "inputs": {"path": "{{csv_data}}"}}
- CSV files without headers have NUMERIC column names: "0", "1", "2", etc. (as strings!)

WORKFLOW PATTERNS:

DATA ANALYSIS:
- Filter then calculate: dataframe_ops(filter) → calculate_statistics
- Example: "sum values >= 100" = filter first, then sum
- Filter creates NEW dataframe with new key (df_0 → df_1)

VISION/IMAGE:
- download_file → analyze_image(task="ocr") → extract text/data
- Use analyze_image for ANY image analysis, NOT call_llm

API:
- fetch_from_api → extract fields using dot notation
- Returns: {"status_code": 200, "data": {...}}

SCRAPING:
- fetch_text → auto render_js_page if needed → extract answer
- If rendered text has answer, STOP (no more tools needed)
- DO NOT call POST/submit endpoints - handled automatically

MULTIMODAL:
- Audio: download_file → transcribe_audio → follow instructions
- Images: download_file → analyze_image → use in calculations
- Combine: audio + vision + data analysis as needed
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
