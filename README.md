# Automated Quiz Solver

An intelligent, LLM-powered system that autonomously solves multi-step quizzes by planning, executing tools, and adapting to different question types.

## 🎯 Overview

This project implements an automated quiz-solving pipeline that:
- Analyzes quiz pages to understand requirements
- Generates multi-step execution plans using LLM reasoning
- Executes diverse tools (data processing, ML, vision, web scraping, etc.)
- Adaptively refines plans based on intermediate results
- Handles complex quiz chains end-to-end

**Key Achievement**: Successfully solves 13 different quiz types including data analysis, machine learning, PDF parsing, OCR, data transformation, and visualization tasks.

## 🏗️ Architecture

### Core Components

```
quiz_solver/
├── main.py                 # FastAPI server and quiz endpoints
├── executor.py             # Main execution orchestrator
├── llm_client.py          # LLM integration (plan generation, reasoning)
├── task_generator.py      # Iterative task generation
├── completion_checker.py  # Smart completion detection
├── parallel_executor.py   # Parallel task execution
├── tool_executors.py      # Tool implementations
├── tools.py               # Tool helper functions
├── tool_definitions.py    # Centralized tool documentation
├── models.py              # Data models
└── cache_manager.py       # Response caching
```

### Design Philosophy

The system follows strict design principles documented in [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md):

1. **LLM-Driven Intelligence**: The LLM interprets instructions and decides operations, not hardcoded logic
2. **Teaching over Patching**: Fix issues by improving prompts, not by adding workaround code
3. **Generic Infrastructure**: Tools work for ANY question type, not demo-specific patterns
4. **Centralized Documentation**: Single source of truth in `tool_definitions.py`

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenRouter API key (for LLM access)

### Installation

```bash
cd quiz_solver
pip install -r requirements.txt
```

### Configuration

Set environment variables:

```bash
export SECRET="your_secret_key"
export OPENROUTER_API_KEY="your_openrouter_key"
```

### Running the Server

```bash
uvicorn main:app --port 8080
```

### Solving a Quiz

```bash
curl -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "secret": "your_secret_key",
    "url": "http://quiz-url.com/quiz"
  }'
```

## 🛠️ Available Tools

The system supports 30+ tools across multiple categories:

### Data Acquisition
- `fetch_from_api` - REST API calls with custom headers
- `download_file` - Binary file downloads (images, PDFs, audio)
- `render_js_page` - JavaScript-rendered page content
- `fetch_text` - Raw text from URLs

### Data Parsing
- `parse_csv` - CSV to DataFrame
- `parse_excel` - Excel files
- `parse_json_file` - JSON to DataFrame
- `parse_pdf_tables` - Extract tables from PDFs
- `parse_html_tables` - HTML table extraction

### Data Processing
- `dataframe_ops` - 20+ operations (filter, pivot, melt, groupby, etc.)
- `calculate_statistics` - Statistical analysis (sum, mean, median, std, etc.)
- `clean_text` - Text normalization
- `extract_patterns` - Regex pattern extraction (emails, URLs, numbers, etc.)

### Machine Learning
- `train_linear_regression` - sklearn linear regression with prediction
- `apply_ml_model` - Generic ML model application

### Multimedia
- `transcribe_audio` - Speech-to-text
- `analyze_image` - Vision AI (OCR, object detection, description)
- `extract_audio_metadata` - Audio file metadata

### Visualization
- `create_chart` - Static charts (matplotlib)
- `create_interactive_chart` - Interactive charts (plotly)
- `make_plot` - Custom plotting

### Utilities
- `call_llm` - LLM reasoning for text analysis
- `zip_base64` - Archive creation

## 🧠 How It Works

### 1. Quiz Analysis

The system renders the quiz page and extracts:
- Text content and instructions
- Images (for vision analysis)
- Audio files (for transcription)
- CSV/data file links
- Submit URL

### 2. Plan Generation

LLM generates an initial execution plan with:
- Task dependencies
- Tool selection
- Input parameters
- Expected outputs (artifacts)

### 3. Iterative Execution

```
┌─────────────────────────────────────────┐
│  Execute Tasks (with parallelization)  │
│  - Resolve artifact references          │
│  - Call tools                           │
│  - Store results as artifacts           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Check Completion (fast + LLM)          │
│  - Statistics calculated?               │
│  - Chart created?                       │
│  - Vision results ready?                │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
       Yes           No
        │             │
        │             ▼
        │   ┌──────────────────────┐
        │   │ Generate Next Tasks  │
        │   │ - What's missing?    │
        │   │ - What operations?   │
        │   └──────┬───────────────┘
        │          │
        │          └──────┐
        │                 │
        ▼                 ▼
   Submit Answer    (Loop back)
```

### 4. Smart Features

**Artifact Resolution**: Dataframe references use template syntax
```python
# LLM generates
{"dataframe_key": "{{df_0}}"}

# Executor resolves
{{df_0}} → artifact → extracts actual registry key "df_3"
```

**Fast Completion Checks**: Deterministic detection of terminal states
- Statistics calculated → complete
- Chart created → complete
- Vision results extracted → complete
- Saves ~4-5 LLM calls per quiz chain

**Parallel Execution**: Independent tasks execute concurrently
```python
Wave 1: [parse_csv, download_file]  # Parallel
Wave 2: [create_chart]               # Depends on Wave 1
```

## 📊 Performance

### Test Suite Results

Successfully solves 13 quiz types:
1. ✅ Literal (direct text answer)
2. ✅ Compute (simple calculation)
3. ✅ Web API (fetch and extract)
4. ✅ Data Analysis (filter + statistics)
5. ✅ ML Challenge (linear regression)
6. ✅ JS Render (extract from JavaScript)
7. ✅ PDF Parse (table extraction)
8. ✅ Vision (OCR from images)
9. ✅ Transform (pivot tables)
10. ✅ Visualize (chart creation)
11. ✅ Multi-step workflows
12. ✅ Complex dependencies
13. ✅ Multimodal data

### Optimizations

- **Completion Check Optimization**: 36.4% skip rate, ~4s saved per chain
- **Parallel Execution**: 2-3x speedup on independent tasks
- **Response Caching**: Prevents redundant file downloads and API calls
- **Artifact Tracking**: Prevents duplicate operations

**Average Performance**: ~17-20 seconds per quiz (including LLM calls)

## 🎓 Key Innovations

### 1. Dataframe Referencing Convention

Solves the "artifact name vs registry key" problem:
- Parse tools store DataFrames with auto-generated keys (`df_0`, `df_1`, etc.)
- LLM references artifacts by name using template syntax: `{{artifact_name}}`
- Executor recursively resolves templates to actual registry keys
- Works for any nesting level in tool parameters

**Why It's Acceptable Infrastructure**:
- Generic pattern (not quiz-specific)
- Enables LLM choice (doesn't force decisions)
- Teachable to LLM through examples
- No business logic

### 2. Teaching-First Approach

When LLM makes mistakes:
1. ❌ **WRONG**: Add code to handle variations
2. ✅ **RIGHT**: Update teaching to clarify correct pattern

Example: Parameter name consistency
```python
# Instead of accepting variations:
df = inputs.get("dataframe") or inputs.get("dataframe_key")  # ❌

# Teach the exact parameter name:
"""
IMPORTANT: Use 'dataframe_key' parameter for train_linear_regression
Example: {"dataframe_key": "{{df_0}}", "feature_columns": [...]}
"""  # ✅
```

### 3. Centralized Tool Documentation

Single source of truth in `tool_definitions.py`:
```python
def get_tool_usage_examples() -> str:
    return """
    AVAILABLE TOOLS:
    - parse_csv(url): Parse CSV → produces dataframe
    - dataframe_ops(op, params): DataFrame operations
      * Example: {"op": "pivot", "params": {"dataframe_key": "{{df_0}}", ...}}
    """

# All prompts import and use this
from tool_definitions import get_tool_usage_examples
```

**Benefits**:
- Update once, propagate everywhere
- Impossible for documentation to drift
- Easy maintenance

## 📚 Documentation

- [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md) - Core design philosophy and anti-patterns
- [TOOL_MAINTENANCE.md](TOOL_MAINTENANCE.md) - Guide for adding/updating tools
- [LLM_FUNCTIONS_DOCUMENTATION.md](LLM_FUNCTIONS_DOCUMENTATION.md) - LLM integration details
- [COMPLETION_OPTIMIZATION_SUMMARY.md](COMPLETION_OPTIMIZATION_SUMMARY.md) - Performance optimizations

## 🧪 Testing

### Run Test Suite

```bash
cd quiz_solver

# Test with local server
python -m pytest test_quizzes.py -v

# Or use the test endpoint
curl http://localhost:8080/solve -X POST -d @test.json
```

### Test Quizzes Available

The server includes 13 test quizzes:
- `/test-quiz/literal` - Simple text extraction
- `/test-quiz/compute` - Basic calculation
- `/test-quiz/web_api` - API integration
- `/test-quiz/data_analysis` - Data processing
- `/test-quiz/ml_challenge` - Machine learning
- `/test-quiz/js_render` - JavaScript rendering
- `/test-quiz/pdf_parse` - PDF table extraction
- `/test-quiz/vision` - OCR/image analysis
- `/test-quiz/transform` - Data transformation
- `/test-quiz/visualize` - Chart creation

## 🔧 Advanced Usage

### Custom Tools

Add new tools in 4 steps:

1. **Define schema** in `tool_definitions.py`:
```python
{
    "type": "function",
    "function": {
        "name": "my_custom_tool",
        "description": "What it does",
        "parameters": {...}
    }
}
```

2. **Add usage examples** in `get_tool_usage_examples()`:
```python
"""
- my_custom_tool(param1, param2): Description
  * Example: {"param1": "value", "param2": 123}
"""
```

3. **Implement** in `tool_executors.py`:
```python
def my_custom_tool(param1: str, param2: int):
    # Implementation
    return result
```

4. **Route** in `executor.py`:
```python
elif tool_name == "my_custom_tool":
    result = my_custom_tool(inputs["param1"], inputs["param2"])
```

See [TOOL_MAINTENANCE.md](TOOL_MAINTENANCE.md) for details.

### Environment Variables

- `SECRET` - Authentication secret for quiz submission
- `OPENROUTER_API_KEY` - OpenRouter API key for LLM access
- `DEFAULT_MODEL` - LLM model (default: "openai/gpt-4o-2024-11-20")

## 🐛 Debugging

### Enable Debug Logging

Uncomment in `main.py`:
```python
logging.basicConfig(level=logging.DEBUG)  # Shows all prompts/responses
```

### Common Issues

**Issue**: LLM generates wrong parameter names
**Solution**: Update teaching in `tool_definitions.py` with explicit examples

**Issue**: Artifact not found
**Solution**: Check artifact resolution logs for `{{artifact}}` pattern usage

**Issue**: Quiz timeout
**Solution**: Increase timeout in `executor.py` or optimize slow tools

## 🤝 Contributing

When adding features:
1. Read [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md) first
2. Follow teaching-first approach
3. Centralize documentation in `tool_definitions.py`
4. Test against multiple quiz types
5. Update this README if adding significant features


## 🙏 Acknowledgments

- OpenRouter for LLM API access
- FastAPI for the web framework
- Tools in Data Science Course for providing the challenge

---

**Built with**: Python, FastAPI, OpenAI API, Pandas, Matplotlib, Playwright, and intelligent LLM orchestration
