#!/usr/bin/env python3
"""Clean up verbose logging in executor.py, keep only essential logs"""

import re

with open('executor.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Patterns to remove (convert to pass or remove line)
remove_patterns = [
    # Task execution details (keep errors/warnings only)
    r'\s+logger\.info\(f"\[TASK_RESULT\] ===== TASK.*',
    r'\s+logger\.info\(f"\[TASK_RESULT\] Tool:.*',
    r'\s+logger\.info\(f"\[TASK_RESULT\] Raw Result:.*',
    r'\s+logger\.info\(f"\[TASK_RESULT\] Result Type:.*',
    r'\s+logger\.info\(f"\[TASK_END\].*',
    r'\s+logger\.info\(f"\[TASK_COMPLETE\].*',
    
    # Artifact storage details
    r'\s+logger\.info\(f"\[ARTIFACT_STORAGE\].*',
    r'\s+logger\.info\(f"\[ARTIFACT_STORED\].*',
    r'\s+logger\.info\(f"\[ARTIFACT_FORMAT\].*',
    r'\s+logger\.info\(f"\[ARTIFACT_LLM\].*',
    r'\s+logger\.info\(f"\[ARTIFACTS_CURRENT\].*',
    
    # Tool execution details (too verbose)
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: download_file.*',
    r'\s+logger\.info\(f"\[TOOL_RESULT\] {task_id}: download_file.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: fetch_text.*',
    r'\s+logger\.info\(f"\[TOOL_RESULT\] {task_id}: fetch_text.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: render_js_page.*',
    r'\s+logger\.info\(f"\[TOOL_RESULT\] {task_id}: render_js_page.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: parse_csv.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: parse_excel.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: parse_json_file.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: parse_html_tables.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: parse_pdf_tables.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: extract_patterns.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: decode_base64.*',
    r'\s+logger\.info\(f"\[TOOL_RESULT\] {task_id}: decode_base64.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: extract_html_text.*',
    r'\s+logger\.info\(f"\[TOOL_RESULT\] {task_id}: extract_html_text.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: parse_csv_data.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: extract_audio_metadata.*',
    r'\s+logger\.info\(f"\[TOOL_RESULT\] {task_id}: extract_audio_metadata.*',
    r'\s+logger\.info\(f"\[TOOL_EXEC\] {task_id}: dataframe_ops.*',
    r'\s+logger\.info\(f"\[TOOL_REQUEST_BODY\].*',
    r'\s+logger\.info\(f"\[TOOL_RESPONSE\].*',
    
    # Parallel execution verbosity
    r'\s+logger\.info\(f"\[PARALLEL\] Wave.*',
    r'\s+logger\.info\(f"\[PARALLEL\] All tasks are I/O-bound.*',
    r'\s+logger\.info\(f"\[PARALLEL\] Mixed tool types.*',
    
    # Plan details (keep summary, remove full JSON dumps)
    r'\s+logger\.info\(f"\[PLAN_JSON\] ===== EXTRACTED JSON START.*',
    r'\s+logger\.info\(f"\[PLAN_JSON\] {plan_json}".*',
    r'\s+logger\.info\(f"\[PLAN_JSON\] ===== EXTRACTED JSON END.*',
    r'\s+logger\.info\(f"\[PLAN_DETAILS\] ===== PARSED PLAN STRUCTURE START.*',
    r'\s+logger\.info\(f"\[PLAN_DETAILS\] {json\.dumps\(plan_obj.*',
    r'\s+logger\.info\(f"\[PLAN_DETAILS\] ===== PARSED PLAN STRUCTURE END.*',
    
    # Execution result dumps
    r'\s+logger\.info\(f"\[EXECUTION_FULL_RESULT\] ===== FULL EXECUTION RESULT START.*',
    r'\s+logger\.info\(f"\[EXECUTION_FULL_RESULT\] {json\.dumps\(execution_result.*',
    r'\s+logger\.info\(f"\[EXECUTION_FULL_RESULT\] ===== FULL EXECUTION RESULT END.*',
    
    # Iteration verbosity
    r'\s+logger\.info\(f"=== Iteration {iteration} ===".*',
    r'\s+logger\.info\(f"Executing {len\(all_tasks\)} tasks".*',
    r'\s+logger\.info\(f"Executing task {task_id}:.*',
    r'\s+logger\.info\("Checking if execution is complete\.\.\.".*',
    r'\s+logger\.info\("Plan has 0 tasks - considered complete".*',
    r'\s+logger\.info\("Answer is ready - stopping iterations".*',
    r'\s+logger\.info\("Generating next batch of tasks\.\.\.".*',
    r'\s+logger\.info\(f"Generated {len\(all_tasks\)} next tasks".*',
    r'\s+logger\.info\("No more tasks generated - stopping iterations".*',
    
    # Page rendering details
    r'\s+logger\.info\(f"\[PAGE_HTML\] HTML length:.*',
    r'\s+logger\.info\(f"\[PAGE_LINKS\] Available links:.*',
    r'\s+logger\.info\(f"\[PAGE_SUBMIT\] Submit URL:.*',
    
    # Artifact extraction details (keep important ones)
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Detected dataframe metadata.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Converted dataframe.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Detected statistics result.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Extracted single statistic.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Extracted \'sum\'.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Extracted \'mean\'.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Detected extract_patterns.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Count question detected.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Value question detected.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] No matches found.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Detected chart result.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Extracted unique_categories.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Detected train_linear_regression.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Extracted prediction.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Detected vision/OCR result.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Extracted vision result.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Extracted result value.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Extracting \'text\' from dict.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Extracted:.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Detected string dict representation.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Extracted text from dict string.*',
    r'\s+logger\.info\(f"\[ARTIFACT_EXTRACTION\] Could not parse dict string.*',
    
    # Artifact resolution
    r'\s+logger\.info\(f"\[ARTIFACT_RESOLVE\].*',
    r'\s+logger\.info\(f"\[ARTIFACT_NESTED\] Extracted field.*',
    
    # Other verbose logs
    r'\s+logger\.info\(f"\[CALCULATE_STATS\].*',
    r'\s+logger\.info\(f"\[CREATE_CHART\].*',
    r'\s+logger\.info\(f"\[CREATE_INTERACTIVE_CHART\].*',
    r'\s+logger\.info\(f"\[TOOL_CORRECT\].*',
    r'\s+logger\.info\("Preparing final answer for submission".*',
    r'\s+logger\.info\(f"Final answer ready for submission:.*',
    r'\s+logger\.info\(f"Submitting answer to.*',
    r'\s+logger\.info\(f"Request body:.*',
    r'\s+logger\.info\(f"Starting execution with iterative plan refinement.*',
]

# Remove matching lines
for pattern in remove_patterns:
    content = re.sub(pattern, '', content)

# Write back
with open('executor.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Cleaned up {len(remove_patterns)} log patterns")
print("Kept: ERROR, WARNING, and essential INFO logs (SUCCESS, FAILURE, MANUAL_OVERRIDE, etc.)")
