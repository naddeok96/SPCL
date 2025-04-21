#!/usr/bin/env python3
"""
split_all_in_one.py

Reads a combined Python file with headers like:

##########################################
# File: some_module.py
##########################################

and splits each section back into its original filename, overwriting those files in-place.
"""

import os
import re
import sys

def split_all_in_one(combined_path):
    # Read the entire combined file
    with open(combined_path, "r") as f:
        text = f.read()

    # This regex matches the header block:
    #   #=====...
    #   # File: filename
    #   #=====...
    # plus the blank line that follows.
    header_re = re.compile(r'^#=+\n# File: (.+)\n#=+\n+', re.MULTILINE)
    parts = header_re.split(text)

    # parts will be like: ["", "utils.py", "<content1>", "replay_buffer.py", "<content2>", ...]
    if len(parts) < 3:
        print("No valid sections found in", combined_path)
        return

    # Iterate over (filename, content) pairs
    for i in range(1, len(parts), 2):
        filename = parts[i].strip()
        content = parts[i+1]

        # Write content back to the original file
        out_path = os.path.join(".", filename)
        with open(out_path, "w") as out_f:
            out_f.write(content)
        print(f"Updated {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 split_all_in_one.py <combined_file>")
        sys.exit(1)

    combined_file = sys.argv[1]
    if not os.path.isfile(combined_file):
        print(f"Error: file '{combined_file}' not found.")
        sys.exit(1)

    split_all_in_one(combined_file)
