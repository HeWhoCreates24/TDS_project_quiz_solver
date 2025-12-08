# Advanced Unseen Test Suite

## Overview

This is a **3-quiz chain designed to test the solver's performance on completely unseen, complex multi-tool workflows**. These quizzes combine multiple tools in ways not seen in the 13-quiz training chain.

## Purpose

The 13-quiz standard chain tests individual tool categories. This advanced chain tests:
- **Complex tool chaining** - Multiple sequential operations
- **Cross-domain workflows** - Combining different tool types
- **Adaptive reasoning** - Novel combinations requiring LLM intelligence
- **Real-world scenarios** - Practical multi-step data processing tasks

## Quiz Breakdown

### Advanced Quiz 1: Multi-Source Data Filtering
**Complexity**: Medium-High  
**Tools Required**: 3-4 tools  
**Workflow**: JSON parsing → Pattern extraction → API fetch → Set difference calculation

**Scenario**: 
- Download JSON file with product reviews
- Extract all email addresses from review text (5 emails total)
- Fetch blacklist from API endpoint (2 blacklisted emails)
- Calculate and return count of non-blacklisted emails

**Expected Answer**: `3` (5 emails - 2 blacklisted = 3 valid)

**Why It's Hard**:
- Requires combining `fetch_from_api` + `extract_patterns` + `call_llm` for set operations
- No direct "filter emails by blacklist" tool
- Must understand set difference concept
- Tests cross-referencing between two data sources

---

### Advanced Quiz 2: Multi-Step Aggregation
**Complexity**: High  
**Tools Required**: 3-4 tools  
**Workflow**: Excel parsing → Filter → GroupBy → Statistical analysis → Comparison

**Scenario**:
- Download Excel file with regional sales data (16 rows, 3 regions, 4 products each)
- Filter rows where revenue > 5000 (keeps 8 rows)
- Group by region and calculate median revenue
- Return the region with highest median revenue

**Expected Answer**: `"West"` (median: ~9050)

**Why It's Hard**:
- Requires Excel parsing (parse_excel)
- Multi-step dataframe operations (filter then groupby)
- Statistical aggregation (median, not just sum/mean)
- Comparative analysis across groups
- Tests understanding of statistical concepts
- Median calculation: North=5100, South=4500, East=5950, **West=9050** ✅

---

### Advanced Quiz 3: Decode → Parse → ML Pipeline
**Complexity**: Very High  
**Tools Required**: 5+ tools  
**Workflow**: JS rendering → Base64 decoding → CSV parsing → ML training → Prediction

**Scenario**:
- Render JavaScript page showing Base64-encoded message
- Decode Base64 to reveal CSV data (hours_studied, exam_score)
- Parse the decoded CSV string into dataframe
- Train linear regression model (hours_studied → exam_score)
- Predict exam score for 8.5 hours of study

**Expected Answer**: `~77.0` (with ±3 tolerance)

**Why It's Hard**:
- **5-tool chain**: render_js_page → (decode step) → parse_csv → train_linear_regression
- Requires text decoding (Base64) - no direct tool, must use call_llm or custom logic
- CSV data is embedded in HTML, not a file URL
- Must parse CSV from string (not URL)
- Linear relationship: y = 7*x + 17.5
- Prediction for x=8.5: y = 7(8.5) + 17.5 = 77.0
- Tests multi-format data transformation pipeline

---

## Running the Advanced Test

### Using Test Script

```bash
# From quiz_solver directory
$body = Get-Content -Path "test_advanced.json" -Raw
$response = Invoke-WebRequest -Uri "http://localhost:8080/solve" `
  -Method Post `
  -Headers @{"Content-Type" = "application/json"} `
  -Body $body
$response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### Expected Performance

**Success Criteria**:
- ✅ All 3 quizzes solved correctly
- ✅ No manual intervention required
- ✅ Adaptive plan generation for unseen workflows
- ✅ Correct tool chaining without hardcoded logic

**Time Estimate**: ~60-90 seconds for full chain

## What Success Proves

If the solver completes this chain successfully:

1. **Generalization**: Works on completely unseen question types
2. **Intelligence**: LLM-driven reasoning, not pattern matching
3. **Tool Composition**: Can combine tools in novel ways
4. **Robustness**: Handles complex multi-step workflows
5. **Production Ready**: Can tackle real-world quiz variations

## Failure Modes to Watch

1. **Quiz 1 Failures**:
   - Can't extract emails from nested JSON text fields
   - Can't perform set difference (blacklist filtering)
   - Returns 5 instead of 3 (doesn't apply blacklist)

2. **Quiz 2 Failures**:
   - Excel parsing issues
   - Groupby/median calculation errors
   - Returns median value instead of region name
   - Wrong region due to calculation error

3. **Quiz 3 Failures**:
   - Can't decode Base64 (no direct tool)
   - Can't parse CSV from string (expects URL)
   - ML model training fails on inline data
   - Prediction error outside tolerance range

## Design Principles Validation

This test validates adherence to [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md):

- ❌ **NO hardcoded workflows** - Each quiz requires adaptive planning
- ❌ **NO demo-specific logic** - Completely new scenarios
- ✅ **LLM-driven decisions** - Must reason about tool combinations
- ✅ **Generic tools** - No quiz-specific tools added
- ✅ **Teaching-first** - Relies on tool documentation, not code patches

## Benchmark Results

Run this suite to establish baseline performance:

```
Quiz Chain: Advanced Unseen Test (3 quizzes)
- Advanced 1 (Multi-Source Filtering): _____ seconds
- Advanced 2 (Multi-Step Aggregation): _____ seconds  
- Advanced 3 (Decode→Parse→ML): _____ seconds
Total Time: _____ seconds
Success Rate: ___/3
```

## Next Steps After Running

1. **If all pass**: System is production-ready for unseen tests
2. **If partial pass**: Analyze failed quizzes for teaching gaps
3. **If all fail**: Review tool documentation and prompts
4. **Document issues**: Add to DESIGN_PRINCIPLES.md if new patterns emerge

---

**This is the ultimate validation test. Good luck!** 🚀
