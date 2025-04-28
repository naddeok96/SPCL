#!/usr/bin/env python3
"""
combine_files.py

This script reads a list of Python file paths, concatenates their contents with clearly
delineated header comments showing the original file names, and then writes the result
to an output file. It also prints the combined code block wrapped in triple backticks,
so that you can copy and paste it easily.
"""

import os

def combine_files(file_paths, output_file):
    """
    Combines the contents of the specified files into a single string with headers.

    Args:
        file_paths (list): List of file paths to combine.
        output_file (str): Path to the output file where the combined code will be saved.

    Returns:
        str: A string containing the combined code from all files.
    """
    combined_code = ""
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist and will be skipped.")
            continue
        filename = os.path.basename(path)
        header = (
            "#" + "=" * 42 + "\n"
            f"# File: {filename}\n"
            "#" + "=" * 42 + "\n\n"
        )
        with open(path, "r") as f:
            content = f.read()
        combined_code += header + content + "\n\n"
    
    with open(output_file, "w") as out:
        out.write(combined_code)
    
    return combined_code

if __name__ == "__main__":
    # Set the list of file paths (update as needed).
    file_paths = [
        "utils.py",
        "replay_buffer.py",
        "rl_agent.py",
        "curriculum.py",
        "curriculum_env.py",
        "generate_dataset.py",
        "off_policy_train.py",
        "on_policy_train.py",
        "config.yaml",

        "analyze_history.py",
        "eval_population.py",
        "init_population.py",
        "merge_history.py",
        "population_utils.py",
        "run_evolution.sh"
    ]
    # Set the output file path.
    output_file = "all_in_one.py"
    
    combined_code = combine_files(file_paths, output_file)
    print(f"Combined code saved to: {output_file}\n")
    
