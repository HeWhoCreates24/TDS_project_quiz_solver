"""Test script for completion detection optimization"""
import sys
sys.path.insert(0, '.')

from completion_checker import get_operation_type, should_check_completion, OPERATION_OVERRIDES

# Test all 24 known tools
print("=" * 80)
print("TOOL CLASSIFICATION TEST")
print("=" * 80)

tools_by_type = {
    'non_terminal': [],
    'intermediate': [],
    'terminal': []
}

for tool_name, expected_type in OPERATION_OVERRIDES.items():
    actual_type = get_operation_type(tool_name)
    tools_by_type[actual_type].append(tool_name)
    
    status = "✅" if actual_type == expected_type else "❌"
    print(f"{status} {tool_name:30s} -> {actual_type:15s} (expected: {expected_type})")

print("\n" + "=" * 80)
print("CLASSIFICATION SUMMARY")
print("=" * 80)
print(f"Non-terminal tools: {len(tools_by_type['non_terminal'])}")
for tool in sorted(tools_by_type['non_terminal']):
    print(f"  - {tool}")

print(f"\nIntermediate tools: {len(tools_by_type['intermediate'])}")
for tool in sorted(tools_by_type['intermediate']):
    print(f"  - {tool}")

print(f"\nTerminal tools: {len(tools_by_type['terminal'])}")
for tool in sorted(tools_by_type['terminal']):
    print(f"  - {tool}")

print("\n" + "=" * 80)
print("SHOULD_CHECK_COMPLETION TEST")
print("=" * 80)

# Test cases
test_cases = [
    # (task, artifacts, expected_should_check, description)
    ({"tool_name": "download_file"}, {}, False, "Non-terminal: download_file"),
    ({"tool_name": "parse_csv"}, {}, False, "Non-terminal: parse_csv"),
    ({"tool_name": "transcribe_audio"}, {}, False, "Non-terminal: transcribe_audio"),
    ({"tool_name": "calculate_statistics"}, {}, True, "Terminal: calculate_statistics"),
    ({"tool_name": "analyze_image"}, {}, True, "Terminal: analyze_image"),
    ({"tool_name": "dataframe_ops"}, {}, True, "Intermediate: dataframe_ops"),
    ({"tool_name": "dataframe_ops"}, {"statistics": {"mean": 42}}, True, "Intermediate with statistics artifact"),
    (None, {}, True, "No task (edge case)"),
]

for task, artifacts, expected, description in test_cases:
    should_check, reason = should_check_completion(task, artifacts)
    status = "✅" if should_check == expected else "❌"
    print(f"{status} {description:45s} -> {should_check} (reason: {reason})")

print("\n" + "=" * 80)
print("SEMANTIC PATTERN TEST (New Tools)")
print("=" * 80)

# Test semantic matching for hypothetical new tools
new_tools = [
    ("fetch_external_data", "non_terminal"),
    ("aggregate_results", "terminal"),
    ("custom_compute_mean", "terminal"),
    ("load_database", "non_terminal"),
    ("generate_report", "terminal"),
    ("filter_records", "non_terminal"),
]

for tool_name, expected in new_tools:
    actual = get_operation_type(tool_name)
    status = "✅" if actual == expected else "❌"
    print(f"{status} {tool_name:30s} -> {actual:15s} (expected: {expected})")

print("\n" + "=" * 80)
print("✅ ALL TESTS COMPLETE")
print("=" * 80)
