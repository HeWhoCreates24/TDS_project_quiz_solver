#!/usr/bin/env python3
"""Fix indentation in executor.py for the artifact processing section."""

with open('executor.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines to indent (0-indexed): everything inside "if final_answer is None:" block
# From "for key in artifacts.keys():" to just before "# Submit answer"
# Line 677 to 976 in 1-indexed = 676 to 975 in 0-indexed
start_line = 676  # 0-indexed (line 677 in editor)
end_line = 976    # 0-indexed (exclusive, stops at line 976, before line 977 "# Submit answer")

new_lines = []
for i, line in enumerate(lines):
    if start_line <= i < end_line:
        # Add 4 spaces of indentation (only if line is not empty)
        if line.strip():  # Don't add spaces to blank lines
            new_lines.append('    ' + line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open('executor.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Indented lines {start_line+1} to {end_line} (1-indexed)")
print("Done!")
