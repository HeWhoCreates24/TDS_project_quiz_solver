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
═══════════════════════════════════════════════════════════════════════════════
║                     WORKFLOW DECISION TREE - START HERE                     ║
═══════════════════════════════════════════════════════════════════════════════

**STEP 1: What type of data are you working with?**

┌─ Tabular data (CSV, Excel, structured JSON)
│  ├─ Has rows and columns? → parse_csv / parse_excel / parse_json_file
│  ├─ Need to TRANSFORM data (rename columns, format values, complex changes)? → call_llm
│  ├─ Need to filter rows? → dataframe_ops(op="filter")
│  ├─ Need column totals? → dataframe_ops(op="sum"/"mean") OR calculate_statistics
│  ├─ Need group totals? → dataframe_ops(op="groupby")
│  └─ Need to reshape structure? → dataframe_ops(op="pivot"/"melt")
│
┌─ Nested JSON / API response (not tabular)
│  ├─ Simple extraction? → call_llm with specific prompt
│  ├─ Complex navigation? → call_llm with path description
│  └─ Want to make it tabular? → parse_json_file (if it's array of objects)
│
┌─ Image with text/numbers
│  └─ Always: download_file → analyze_image(task="ocr")
│
┌─ Audio file
│  └─ Always: download_file → transcribe_audio
│     └─ If transcription has numbers/values: → extract_patterns or call_llm
│
┌─ HTML page
│  ├─ Static content? → fetch_text
│  ├─ Dynamic (JavaScript)? → render_js_page
│  ├─ Has Base64 data? → render_js_page → extract_html_text(selector) → decode_base64
│  └─ Has tables? → render_js_page → parse_html_tables
│
└─ PDF document
   └─ Has tables? → download_file → parse_pdf_tables

**STEP 2: Examine metadata BEFORE choosing next operation**

After parsing, you get metadata like:
{
  "dataframe_key": "df_0",
  "columns": ["ID", "Name", "Value"],  ← Use EXACT names (case-sensitive!)
  "shape": [100, 3],
  "sample_data": [[1, "Alice", 100], ...]
}

- Columns tell you EXACT names to use ("ID" not "id")
- Sample shows data types and patterns
- Shape tells you rows × columns

**STEP 3: Choose operation based on what you need**

┌─ I need: A subset of rows
│  └─ Use: dataframe_ops(op="filter", params={"condition": "Value >= 100"})
│
┌─ I need: Specific columns only
│  └─ Use: dataframe_ops(op="select", params={"columns": ["ID", "Name"]})
│
┌─ I need: Total/average of a column
│  └─ Use: dataframe_ops(op="sum"/"mean", params={"column": "Value"})
│
┌─ I need: Calculated total (e.g., sum of Quantity × Price for each row)
│  └─ Use: dataframe_ops(op="eval", params={"expression": "(Quantity * UnitPrice).sum()"})
│
┌─ I need: Per-group totals (e.g., sum by category)
│  └─ Use: dataframe_ops(op="groupby", params={"by": ["Category"], "aggregation": {"Value": "sum"}})
│
┌─ I need: Row with max value in column X, return value from column Y
│  └─ Use: dataframe_ops(op="idxmax", params={"value_column": "X", "label_column": "Y"})
│
└─ I need: Rows as columns (pivot table)
   └─ Use: dataframe_ops(op="pivot", params={"index": "row_var", "columns": "col_var", "values": "data"})

**STEP 4: Reference artifacts correctly**

- Parse creates artifact "df_0" with metadata {"dataframe_key": "df_2", ...}
- In next tool, use: {"dataframe_key": "{{df_0}}"} ← Template markers required!
- Executor resolves {{df_0}} → extracts "df_2" → passes to tool
- Each operation creates NEW artifact (df_0 → filter → df_1)

════════════════════════════════════════════════════════════════════════════════

AVAILABLE TOOLS AND USAGE:

Web & Data Fetching:
- render_js_page(url): Render webpage with JavaScript, extract text/links/code
- fetch_text(url): Fetch raw text content from URL
- fetch_from_api(url, method, headers, body): Call REST APIs with custom headers
  * Returns: {"status_code": 200, "data": {...}} - JSON response object
  * NOT a dataframe - use call_llm for simple queries or parse_json_file to convert
- download_file(url): Download binary files (images, audio, PDFs, etc.)

File Parsing:
- parse_csv(path OR url): Parse CSV file → produces {"dataframe_key": "df_0"}
- parse_csv_data(csv_content, delimiter): Parse CSV string content → produces {"dataframe_key": "df_X"}
  * Use after decode_base64 or when you have CSV as a string
  * Example: decode_base64 → parse_csv_data(decoded_text)
- parse_excel(path): Parse Excel files → produces dataframe
- parse_json_file(path): Parse JSON to dataframe → produces {"dataframe_key": "df_X"}
  * Use this to convert JSON to dataframe for dataframe_ops
  * For simple JSON queries, use call_llm instead
- parse_html_tables(path_or_html): Extract HTML tables
- parse_pdf_tables(path, pages): Extract tables from PDFs

Text Processing:
- clean_text(text, remove_special_chars, normalize_whitespace): Clean/normalize text
- extract_html_text(html, selector): Extract text content from HTML
  * Use to extract Base64 strings, codes, or other text from HTML pages
  * selector: Optional CSS selector to target specific elements
  * Returns: {"text": "extracted content"}
- decode_base64(encoded_text): Decode Base64 encoded text/data
  * Input should be the Base64 string itself (extract from HTML first if needed using extract_html_text)
  * Returns: {"decoded_text": "..."}
  * Example workflow: render_js_page → extract_html_text(html, "code") → decode_base64 → parse_csv_data
- extract_patterns(text, pattern_type, custom_pattern): Extract patterns from text
  * pattern_type: "email", "url", "phone", "date", "number" (SINGULAR, not "emails")
  * Returns: {"matches": [...], "count": N}
  * Example: {"text": "Contact support@example.com", "pattern_type": "email"}
  * Use to extract specific values from natural language text (e.g., numbers from "The answer is 42")

Data Transformation:
- dataframe_ops(op, params): All DataFrame operations including transformations
  * Works on dataframes (from parse_csv, parse_excel, parse_json_file)
  * JSON from fetch_from_api is a dict, not a dataframe - use call_llm for dict queries
  * Dataframe reference: Use params.dataframe_key = "{{artifact_name}}" with template markers
  * Structure: {"op": "pivot", "params": {"dataframe_key": "{{df_0}}", "index": "...", "columns": "...", "values": "..."}}
  * Row operations: filter, sum, mean, count, select
  * Aggregation: groupby (with 'by' and 'aggregation' parameters)
  * Shape operations: pivot, melt, transpose
  * Sorting/slicing: sort (by column), head (first N rows), tail (last N rows)
  * Row extraction: idxmax, idxmin (find row with max/min value and extract label)
  
  ═══════════════════════════════════════════════════════════════════════════════
  ║ DATAFRAME COLUMN HANDLING - READ THIS FIRST BEFORE USING ANY DATAFRAME OPS ║
  ═══════════════════════════════════════════════════════════════════════════════
  
  **STEP 1: EXAMINE DATAFRAME METADATA**
  After parsing (parse_csv, parse_excel, parse_json_file), you receive metadata:
  {
    "dataframe_key": "df_0",
    "columns": ["ID", "Name", "JoinDate", "Value"],  ← EXACT column names
    "sample_data": [[1, "Alice", "2024-01-15", 100], [2, "Bob", "2024-02-20", 200]]
  }
  
  **STEP 2: USE EXACT COLUMN NAMES (CASE-SENSITIVE)**
  - Column names are CASE-SENSITIVE: "Value" ≠ "value" ≠ "VALUE"
  - Use the EXACT names from the 'columns' field in metadata
  - Example: If columns=["ID", "Name"], use "ID" not "id" or "Id"
  
  **STEP 3: CHOOSE THE RIGHT OPERATION**
  
  A) FILTERING DATA (when you need a subset of rows)
     {"op": "filter", "params": {
       "dataframe_key": "{{df_0}}",
       "condition": "Value >= 100"  ← Use EXACT column name from metadata
     }}
     - Operators: >=, <=, >, <, ==, !=
     - Format: "ColumnName operator value" (e.g., "Value >= 100", "Name == 'Alice'")
     - Creates NEW dataframe (df_0 → filter → df_1)
     - DO NOT filter for null/empty unless explicitly asked - work with data as-is
  
  B) SELECTING COLUMNS (when you need specific columns)
     {"op": "select", "params": {
       "dataframe_key": "{{df_0}}",
       "columns": ["ID", "Value"]  ← Array of EXACT column names
     }}
     - Returns only specified columns
     - Preserves all rows
  
  C) AGGREGATING VALUES (when you need totals/averages)
     {"op": "sum", "params": {"dataframe_key": "{{df_0}}", "column": "Value"}}
     {"op": "mean", "params": {"dataframe_key": "{{df_0}}", "column": "Value"}}
     - Returns single numeric result
  
  D) GROUPING AND AGGREGATING (when you need per-category totals)
     ════════════════════════════════════════════════════════════════
     ║ CRITICAL: TWO FORMATS - USE THE RIGHT ONE!                   ║
     ════════════════════════════════════════════════════════════════
     
     FORMAT 1 - SIMPLE (Keep Original Column Names):
     {"op": "groupby", "params": {
       "dataframe_key": "{{df_0}}",
       "by": ["customer_id"],
       "aggregation": {"amount": "sum"}  ← Key is EXISTING column name
     }}
     → Result columns: ["customer_id", "amount"]
     → Use when you DON'T need to rename the result column
     
     FORMAT 2 - NAMED (Rename Result Columns):
     {"op": "groupby", "params": {
       "dataframe_key": "{{df_0}}",
       "by": ["customer_id"],
       "aggregation": {"total": ["amount", "sum"]}  ← Array: [source_col, function]
     }}
     → Result columns: ["customer_id", "total"]
     → Use when you NEED to rename (e.g., 'amount' → 'total')
     
     ════════════════════════════════════════════════════════════════
     ║ COMMON MISTAKE - WRONG FORMAT USAGE:                         ║
     ════════════════════════════════════════════════════════════════
     Given: orders.csv with columns ['customer_id', 'order_date', 'amount']
     
     ❌ WRONG - Format 1 with non-existent column:
        {"aggregation": {"total": "sum"}}
        → ERROR: "Column(s) ['total'] do not exist"
        → You tried to aggregate a column called 'total' but it doesn't exist!
     
     ✅ CORRECT - Format 2 to create 'total' column:
        {"aggregation": {"total": ["amount", "sum"]}}
        → Aggregates 'amount' (which exists) and names result 'total'
     
     ✅ ALSO CORRECT - Format 1 to keep 'amount' name:
        {"aggregation": {"amount": "sum"}}
        → Aggregates 'amount' and keeps the name 'amount'
     
     ════════════════════════════════════════════════════════════════
     
     - Aggregation functions: "count", "sum", "mean", "min", "max", "median"
     - Returns dataframe with grouped columns as regular columns (not index)
     - To get top N: Chain operations → groupby → sort → head
       Example: groupby → sort(by='total', ascending=False) → head(n=3)
  
  **COMMON MISTAKES TO AVOID:**
  ❌ Using lowercase when column is uppercase: "value" when it's "Value"
  ❌ Using SQL-style conditions: "WHERE Value > 100" (just use "Value > 100")
  ❌ Using unsupported operators: "is", "is not", "not", "AND", "OR"
  ❌ Filtering nulls unnecessarily: Only filter if task explicitly requires it
  ❌ Hardcoding registry keys: Use {{df_0}} not "df_2"
  
  E) SORTING DATA (when you need ordered results)
     {"op": "sort", "params": {
       "dataframe_key": "{{df_0}}",
       "by": "Value",  ← Column name or list of columns
       "ascending": False  ← False for descending (largest first)
     }}
     - Returns dataframe sorted by specified column(s)
     - ascending: True (smallest first) or False (largest first)
  
  F) TAKING FIRST/LAST ROWS (when you need top N or bottom N)
     {"op": "head", "params": {"dataframe_key": "{{df_0}}", "n": 3}}  ← First 3 rows
     {"op": "tail", "params": {"dataframe_key": "{{df_0}}", "n": 5}}  ← Last 5 rows
     - Use after sort to get top/bottom N by value
     - Example: sort(by='amount', ascending=False) then head(n=3) = top 3 by amount
  
  **WORKFLOW PATTERN:**
  1. Parse data → Examine columns field in metadata
  2. Choose operation based on task (filter/select/groupby/sort/head/aggregate)
  3. Use EXACT column names from metadata
  4. Reference dataframe with {{artifact_name}}
  
  * Groupby example: {"op": "groupby", "params": {"dataframe_key": "{{df_0}}", "by": ["region"], "aggregation": {"revenue": "median"}}}
    - Parameter 'by': Array of column names to group by
    - Parameter 'aggregation': String like "count" OR dict like {"revenue": "median"}. Default: "count"
    - Result has grouped columns as regular columns (index reset automatically)
  * Idxmax example: {"op": "idxmax", "params": {"dataframe_key": "{{df_2}}", "value_column": "revenue", "label_column": "region"}}
    - Finds the row with max value in 'value_column'
    - Returns the value from 'label_column' in that row (e.g., region name)
    - Use when you need the LABEL (e.g., "West") not the VALUE (e.g., 9050)
  * Pivot example: {"op": "pivot", "params": {"dataframe_key": "{{df_0}}", "index": "category", "columns": "month", "values": "sales"}}
  * Sum example: {"op": "sum", "params": {"dataframe_key": "{{df_1}}", "column": "sales"}}
  * Select example: {"op": "select", "params": {"dataframe_key": "{{df_0}}", "columns": ["ID", "Name", "Value"]}}
  * Filter creates NEW dataframe: df_0 → filter → df_1

- calculate_statistics(dataframe, stats, columns): Calculate sum, mean, median, std, etc.
  * Works on dataframes (dataframe_key like "df_0")
  * Dataframe reference: Use dataframe = "{{artifact_name}}" with template markers
  * Parameter name: 'dataframe' (top-level parameter specifying which dataframe)
  * Example: {"dataframe": "{{df_0}}", "stats": ["sum"], "columns": ["sales"]}
  * Use for statistical analysis on columns

Machine Learning:
- train_linear_regression(dataframe_key, feature_columns, target_column, predict_x): sklearn regression
  * Dataframe reference: Use dataframe_key = "{{artifact_name}}" with template markers
  * Parameter name: 'dataframe_key' (top-level parameter specifying which dataframe)
  * Example: {"dataframe_key": "{{df_0}}", "feature_columns": ["x"], "target_column": "y", "predict_x": {"x": 50}}
  * Returns model coefficients and optional prediction

- apply_ml_model(dataframe, model_type, kwargs): Apply ML models

Multimedia:
- transcribe_audio(audio_path): Speech-to-text transcription
  * Returns: {"text": "transcribed content"}
  * Transcription output may contain natural language - use extract_patterns or call_llm to parse specific values
  * Example: transcription returns "The answer is 42" → use extract_patterns(text, "number") to get "42"
- analyze_image(image_path, task): Vision AI for OCR, description, object detection, classification
  * task="ocr" extracts text/numbers from images
  * Use for ANY image analysis, NOT call_llm (call_llm cannot process images)
- extract_audio_metadata(path): Get audio duration, sample rate, etc.

Visualization:
- create_chart(dataframe, chart_type, x_col, y_col, title, output_path): Create static charts
  * Dataframe reference: Use dataframe = "{{artifact_name}}" with template markers
  * Parameter name: 'dataframe' (the dataframe_key string, e.g., "df_0")
  * output_path is optional - omit this parameter when chart saving is not needed
  * Returns: {"chart_path": "path/to/chart.png", "unique_categories": N}
  * The unique_categories field contains the count of unique values in x_col
  * Use this when quiz asks "how many categories in the chart"
  * Example (no saving): {"dataframe": "{{df_0}}", "chart_type": "bar", "x_col": "category", "y_col": "sales"}
  * Example (with saving): {"dataframe": "{{df_0}}", "chart_type": "bar", "x_col": "category", "y_col": "sales", "output_path": "/absolute/path/to/chart.png"}
- create_interactive_chart(dataframe, chart_type, x_col, y_col, title): Plotly interactive charts
- make_plot(spec): Custom plotting with detailed specs

Utilities:
- call_llm(prompt, system_prompt, max_tokens, temperature): Send text to LLM for analysis
  * Use for simple JSON queries (find max, filter, extract values) when working with text/JSON data
  * Example: Find product with highest price from JSON → use call_llm
  * For structured dataframes, dataframe_ops or calculate_statistics provide specialized operations
  
  ═══════════════════════════════════════════════════════════════════════════════
  ║              CALL_LLM PROMPT ENGINEERING - CRITICAL PATTERNS                ║
  ═══════════════════════════════════════════════════════════════════════════════
  
  **PRINCIPLE: Be HYPER-SPECIFIC about output format**
  
  The LLM will return EXACTLY what you describe. If you're vague, you'll get:
  - JSON objects when you wanted plain strings
  - Explanations when you wanted just the answer
  - Multiple values when you wanted one
  
  **PATTERN 1: Extracting Single Values**
  ❌ BAD: "Find the product with highest price and return its id"
     Result: {"id": "P004", "price": 299.99} ← Unwanted JSON wrapper
  
  ✅ GOOD:
     prompt: "Here is product data: {{json_data}}. Extract the 'id' field from the product with the highest 'price' value. Return ONLY the id value itself (like 'P004'), not JSON, not an object, not a sentence - just the raw id string."
     system_prompt: "You extract data values. Return ONLY the requested value with no formatting, no JSON, no explanation."
     Result: "P004" ← Perfect!
  
  **PATTERN 2: Numeric Calculations**
  ❌ BAD: "Calculate the total revenue"
     Result: "The total revenue is $15,432.50" ← Unwanted text
  
  ✅ GOOD:
     prompt: "Calculate the sum of all 'price' values in this data: {{json_data}}. Return ONLY the numeric result (e.g., 15432.50), no dollar signs, no commas, no text."
     system_prompt: "You are a calculator. Return ONLY the numeric answer."
     Result: "15432.50" ← Perfect!
  
  **PATTERN 2b: Calculations with Explicit Formulas**
  ❌ BAD: "Calculate F1 score for each run"
     Result: Uses wrong formula or makes arithmetic errors
  
  ✅ GOOD - CRITICAL: When quiz provides a formula, USE IT EXACTLY:
     prompt: "Given this data: {{{{data}}}}
             
             For each run, calculate metric using THIS EXACT FORMULA:
             F1 = 2*tp / (2*tp + fp + fn)
             
             Example calculation (show your work):
             - If tp=9, fp=1, fn=1: F1 = 2*9 / (2*9 + 1 + 1) = 18/20 = 0.9
             
             Steps:
             1. For EACH run, calculate F1 for EACH label using the formula above
             2. Average F1 scores across labels to get macro-F1
             3. Round macro-F1 to 4 decimal places
             
             Find the run with HIGHEST macro-F1.
             
             Return ONLY this JSON (no explanations, no steps in output):
             {{'run_id': 'runX', 'macro_f1': 0.XXXX}}
             
             CRITICAL: Use the EXACT formula provided. Do NOT use precision/recall formulas."
     system_prompt: "Follow formulas EXACTLY as given. Show arithmetic step-by-step in your thinking, but return ONLY the final JSON result."
     Result: {{'run_id': 'runC', 'macro_f1': 0.8175}} ← Perfect!
  
  **KEY: ALWAYS specify exact output format - JSON structure, field names, no explanations**
  
  **KEY: ALWAYS specify exact output format - JSON structure, field names, no explanations**
  
  **PATTERN 3: Finding Maximum/Minimum**
  ✅ TEMPLATE:
     prompt: "From this JSON array: {{data}}, find the item with the maximum '{{field}}' value and return its '{{target_field}}' field. Return ONLY that value, nothing else."
     system_prompt: "Extract the requested field value. Return it raw with no JSON wrapping."
  
  **PATTERN 4: Filtering/Counting**
  ✅ TEMPLATE:
     prompt: "Count how many items in this array have '{{field}}' > {{threshold}}: {{data}}. Return ONLY the count as a number."
     system_prompt: "Return ONLY the count as a plain number."
  
  **PATTERN 5: Complex JSON Navigation**
  ✅ TEMPLATE:
     prompt: "Navigate this nested JSON: {{json}}. Extract the value at path {{path}}. Return ONLY that value with no additional formatting."
     Example: "Extract data.users[0].email from {{json}}. Return ONLY the email address."
  
  **KEY RULES:**
  1. Always say "Return ONLY [what you want]" in the prompt
  2. Explicitly list what NOT to include: "no JSON", "no explanation", "no text"
  3. Give an example format: "(e.g., 'P004')" or "(e.g., 42.5)"
  4. Use system_prompt to reinforce: "You return raw values only"
  5. For numbers: Specify "no commas, no dollar signs, no units"
  6. For strings: Specify "no quotes, no brackets"
  
  **WHEN TO USE call_llm vs DATAFRAME_OPS:**
  - Use call_llm: For JSON dicts, nested objects, text data, simple queries, DATA TRANSFORMATION TASKS
  - Use dataframe_ops: For tabular data with rows/columns after parse_csv/parse_json_file
  
  **DATA TRANSFORMATION USE CASES:**
  When task requires transforming data structure, format, or content:
  - Renaming columns (e.g., "ID" → "id", camelCase → snake_case)
  - Reformatting values (e.g., dates "01/15/24" → "2024-01-15", strings → integers)
  - Complex reshaping beyond pivot/melt
  - Combining multiple transformation steps
  
  **Pattern for data transformation:**
  1. Parse CSV → Get dataframe with sample_data in artifact
  2. Use call_llm referencing the dataframe artifact - executor will inject full data
  3. LLM transforms the data and returns result
  
  **CRITICAL: How to access dataframe data in call_llm:**
  When you need to transform dataframe data, reference the artifact in the prompt.
  The prompt should ask call_llm to work with "the dataframe from {{df_0}}".
  The executor will automatically include the dataframe sample in the context.
  
  **Example - Column renaming and date formatting:**
  ```
  {
    "tool_name": "call_llm",
    "inputs": {
      "prompt": "I have a dataframe with this structure: {{df_0}}. Transform ALL the data as follows: 1) Convert column names to snake_case (ID→id, Name→name, Joined→joined, Value→value), 2) Standardize ALL dates to YYYY-MM-DD format (handle MM/DD/YY, YYYY-MM-DD, and 'D Mon YYYY' formats), 3) Ensure all numeric values are integers, 4) Sort by id ascending. Return ONLY the complete JSON array with all rows, no explanation.",
      "system_prompt": "You transform data structures. Return only valid JSON arrays with no additional text."
    },
    "produces": ["transformed_data"]
  }
  ```
  
  **KEY POINTS:**
  - Say "ALL the data" or "complete data" to avoid partial responses
  - Reference {{df_0}} in prompt - metadata will be injected
  - Be explicit about handling ALL date formats present in the data
  - Request "complete JSON array with all rows" to prevent truncation
  - System prompt reinforces: return ONLY JSON, no explanation
  
  **When dataframe_ops IS appropriate:**
  - Filtering rows: "Keep only rows where Value > 100"
  - Selecting columns: "Keep only ID and Name columns"
  - Aggregating: "Sum all values", "Calculate mean by category"
  - Grouping: "Total sales by region"
  - Pivoting: "Reshape categories into columns"
  
  **Key distinction:**
  - call_llm: Transforms data CONTENT and STRUCTURE (renaming, formatting, complex logic)
  - dataframe_ops: Operates on EXISTING structure (filter, select, aggregate, reshape shape)

- zip_base64(paths): Create zip archives
- geospatial_analysis(dataframe, analysis_type, kwargs): Distance, geocoding, spatial joins
- generate_narrative(dataframe, summary_stats): Generate natural language from data

CRITICAL PATTERNS:
- Images → download_file + analyze_image (task="ocr")
- Audio → download_file + transcribe_audio
- CSV/Excel analysis → parse_csv/parse_excel + dataframe_ops/calculate_statistics
- Base64 in HTML → render_js_page + extract_html_text(html, selector) + decode_base64 + parse_csv_data
  * Extract the Base64 string from HTML using a CSS selector to target the encoded text element
  * Common selectors for encoded data: "code", "pre", "#message-container code", ".encoded"
  * Example: extract_html_text(html="{{html_content}}", selector="code")
  * Using a selector extracts only the target element; omitting it extracts all text from the page
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

1. DATA TRANSFORMATION (renaming, formatting, complex changes):
   Pattern: parse_csv → call_llm with transformation instructions
   Example task: "Normalize CSV: convert columns to snake_case, standardize dates to ISO-8601, sort by id"
   Solution: parse_csv gets data → call_llm transforms structure/format → return JSON
   Key point: Use call_llm for content/format changes, not dataframe_ops
   
2. API FETCH WITH NESTED FIELD:
   Tool: fetch_from_api (NOT "call_api")
   Pattern: fetch_from_api returns {"status_code": 200, "data": {...}}
   Example task: Extract 'secret_code' from API response
   Solution: Use fetch_from_api, reference result with dot notation
   Key point: Artifact naming must match between produces and final_answer_spec
   
3. VISUALIZATION + COUNT:
   Tool: create_chart (NOT dataframe_ops count)
   Pattern: create_chart returns {"chart_path": "...", "unique_categories": N}
   Example task: "Create bar chart and count how many categories"
   Solution: Use create_chart, reference .unique_categories field
   Key point: The unique_categories field automatically contains the count
   
4. DATA TRANSFORMATION:
   Tool: dataframe_ops with op="pivot"
   Pattern: Pivot restructures data shape (rows → columns)
   Example task: "Pivot by category and month, sum January sales"
   Solution: First pivot with {"op": "pivot", "params": {"dataframe_key": "{{df_0}}", ...}}
   Then calculate_statistics on result: {"dataframe": "{{pivoted}}", "stats": ["sum"], ...}
   Key point: After pivot, month names become column names

5. MACHINE LEARNING WORKFLOW:
   Tool: train_linear_regression
   Pattern: Parse CSV → Train model → Get prediction
   Example: {"dataframe_key": "{{df_0}}", "feature_columns": ["x"], "target_column": "y", "predict_x": {"x": 50}}
   Key point: Use {{artifact_name}} to reference the parsed dataframe

PARAMETER NAMING CONVENTIONS:
- dataframe_ops: Use {"op": "...", "params": {"dataframe_key": "{{artifact}}", ...}}
  * CRITICAL: Parameter is "op" NOT "operation"
  * CRITICAL: Dataframe identifier goes in params.dataframe_key with template markers
  * Works for ALL operations: filter, sum, mean, count, select, groupby, sort, head, tail, pivot, melt, transpose, idxmax, idxmin
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
- parse_csv expects a direct URL or file path to fetch data from:
  * Example: {"tool_name": "parse_csv", "inputs": {"url": "http://example.com/data.csv"}}
  * For CSV content already in memory, use parse_csv_data instead: {"csv_content": "{{decoded_text}}"}
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
                "name": "parse_csv_data",
                "description": "Parse CSV content from a string (e.g., decoded Base64, API response) into a pandas DataFrame",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "csv_content": {
                            "type": "string",
                            "description": "CSV content as a string"
                        },
                        "delimiter": {
                            "type": "string",
                            "description": "CSV delimiter character (default: comma)"
                        }
                    },
                    "required": ["csv_content"]
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
                "name": "extract_html_text",
                "description": "Extract text content from HTML. Provide a CSS selector to target specific elements (e.g., 'code', 'pre' for Base64 strings). Without a selector, extracts all visible text from the page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "html": {
                            "type": "string",
                            "description": "HTML content to extract text from"
                        },
                        "selector": {
                            "type": "string",
                            "description": "CSS selector to target specific elements. For Base64/encoded data, use 'code' or 'pre'. Examples: 'code', '#message-container code', 'pre', '.encoded-text'"
                        }
                    },
                    "required": ["html"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "decode_base64",
                "description": "Decode Base64 encoded text. Use this for encoded messages, data, or content that needs to be decoded.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "encoded_text": {
                            "type": "string",
                            "description": "Base64 encoded string to decode"
                        }
                    },
                    "required": ["encoded_text"]
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
                "description": "Perform DataFrame operations on parsed tabular data. CRITICAL: Column names are CASE-SENSITIVE - always use exact names from dataframe metadata. Operations: filter rows by conditions, calculate aggregations (sum/mean/count), select specific columns, group by categories with aggregation, sort by columns, take first/last N rows (head/tail), reshape data (pivot/melt/transpose), or find rows with max/min values (idxmax/idxmin). Each operation creates a new DataFrame artifact.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Operation to perform. Choose based on goal: 'filter' for row subsets, 'select' for column subsets, 'sum'/'mean' for single values, 'groupby' for per-category aggregation, 'pivot' for reshaping, 'idxmax'/'idxmin' for finding extreme values, 'eval' for calculated expressions like (Column1 * Column2).sum()",
                            "enum": ["filter", "sum", "mean", "count", "select", "groupby", "pivot", "melt", "transpose", "sort", "head", "tail", "idxmax", "idxmin", "eval"]
                        },
                        "params": {
                            "type": "object",
                            "description": "Parameters for the operation. Always include 'dataframe_key' using template syntax: {{artifact_name}}",
                            "properties": {
                                "dataframe_key": {
                                    "type": "string",
                                    "description": "Reference to parsed dataframe using template markers. Format: {{artifact_name}} where artifact_name is from parse tool output (e.g., {{df_0}}). Do NOT use raw registry keys or omit template markers."
                                },
                                "condition": {
                                    "type": "string",
                                    "description": "FOR FILTER ONLY. Complete condition with EXACT column name (case-sensitive!), comparison operator, and value. Format: 'ColumnName operator value'. Valid operators: >= <= > < == !=. Examples: 'Temperature > 25', 'ID >= 100', 'Name == Alice'. Use column names EXACTLY as shown in dataframe metadata."
                                },
                                "column": {
                                    "type": "string",
                                    "description": "FOR SUM/MEAN ONLY. Column name to aggregate. Must match EXACT column name from metadata (case-sensitive). Example: If metadata shows 'Revenue', use 'Revenue' not 'revenue'."
                                },
                                "columns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "FOR SELECT ONLY. Array of column names to keep. Must use EXACT column names from metadata (case-sensitive). Example: If metadata shows ['ID', 'Name', 'Value'], use ['ID', 'Name'] not ['id', 'name']."
                                },
                                "index": {
                                    "type": "string",
                                    "description": "FOR PIVOT ONLY. Column name to use as row labels in pivoted result. Must match exact column name from metadata."
                                },
                                "columns": {
                                    "type": "string",
                                    "description": "FOR PIVOT ONLY. Column whose unique values become column headers in pivoted result. Must match exact column name from metadata."
                                },
                                "values": {
                                    "type": "string",
                                    "description": "FOR PIVOT ONLY. Column containing the data values to fill the pivot table. Must match exact column name from metadata."
                                },
                                "id_vars": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "FOR MELT ONLY. Columns to keep as identifier variables (not unpivoted). Use exact column names from metadata."
                                },
                                "value_vars": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "FOR MELT ONLY. Columns to unpivot into variable/value pairs. Use exact column names from metadata."
                                },
                                "var_name": {
                                    "type": "string",
                                    "description": "FOR MELT ONLY. Name for the new 'variable' column that will contain unpivoted column names. Optional, defaults to 'variable'."
                                },
                                "value_name": {
                                    "type": "string",
                                    "description": "FOR MELT ONLY. Name for the new 'value' column that will contain unpivoted values. Optional, defaults to 'value'."
                                },
                                "by": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "FOR GROUPBY ONLY. Column names to group by. Must use EXACT column names from metadata. Example: ['Region', 'Category']. Result will have these columns plus aggregated columns."
                                },
                                "aggregation": {
                                    "description": "FOR GROUPBY ONLY. Aggregation specification. Can be: (1) String like 'count', 'sum', 'mean', 'min', 'max', 'median' to apply to all numeric columns, OR (2) Dict for column-specific aggregation. Dict format supports two styles: SIMPLE: {'column_name': 'function'} keeps original column name, example {'amount': 'sum'} → result has 'amount' column. NAMED: {'new_name': ['existing_column', 'function']} renames aggregated column, example {'total': ['amount', 'sum']} → result has 'total' column. CRITICAL: For simple format, dict keys MUST be existing columns. For named format, array's first element must be existing column. Use JSON arrays, not Python tuples. Defaults to 'count' if omitted."
                                },
                                "by": {
                                    "description": "FOR SORT ONLY. Column name (string) or list of column names to sort by. Must use EXACT column names from metadata. Example: 'amount' or ['customer_id', 'amount']."
                                },
                                "ascending": {
                                    "type": "boolean",
                                    "description": "FOR SORT ONLY. Sort direction. True for ascending (smallest first), False for descending (largest first). Default: True."
                                },
                                "n": {
                                    "type": "integer",
                                    "description": "FOR HEAD/TAIL ONLY. Number of rows to return. Default: 5. Use head to get first N rows (top N after sorting), tail to get last N rows."
                                },
                                "value_column": {
                                    "type": "string",
                                    "description": "FOR IDXMAX/IDXMIN ONLY. Column to find maximum/minimum value in. Must use exact column name from metadata. Example: Find row with highest 'Revenue' → use 'Revenue'."
                                },
                                "label_column": {
                                    "type": "string",
                                    "description": "FOR IDXMAX/IDXMIN ONLY. Column to extract value from in the row with max/min. Returns the label/name, not the max/min value itself. Example: Find region with highest revenue → value_column='Revenue', label_column='Region' → returns region name like 'West'."
                                },
                                "expression": {
                                    "type": "string",
                                    "description": "FOR EVAL ONLY. Pandas expression to evaluate on the DataFrame. Supports column operations and aggregations. Use for calculated totals like row-wise products summed. Example: '(Quantity * UnitPrice).sum()' calculates quantity×price for each row then sums total. Must use EXACT column names from metadata (case-sensitive). Returns single value or Series."
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
