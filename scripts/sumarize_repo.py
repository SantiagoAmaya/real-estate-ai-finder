#!/usr/bin/env python3

import os
import ast
import yaml  # Requires 'pip install pyyaml'
import sys

# --- Configuration ---

# Add any directories you want to skip
IGNORE_DIRS = {
    '.git',
    '__pycache__',
    'node_modules',
    '.vscode',
    'venv',
    '.venv',
    'dist',
    'build',
    'docs' # Often good to skip, or remove if you want to parse it
}

# --- File Parsers ---

def get_python_docstring(filepath):
    """Safely parses a Python file and returns its module-level docstring."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree)
        if docstring:
            # Return just the first line for a clean summary
            return docstring.strip().splitlines()[0]
        return None
    except Exception as e:
        return f"[Error parsing Py: {e.__class__.__name__}]"


def get_md_summary(filepath):
    """Returns the first non-empty, non-header line from a Markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    return stripped_line
        return None  # No summary found
    except Exception as e:
        return f"[Error reading MD: {e.__class__.__name__}]"


def get_yaml_description(filepath):
    """Parses a YAML file and looks for a 'description' or 'summary' key."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if isinstance(data, dict):
            # Try to find a common description key
            for key in ['description', 'summary', 'info']:
                if key in data and isinstance(data[key], str):
                    return data[key].strip().splitlines()[0]
        return None
    except Exception as e:
        return f"[Error parsing YAML: {e.__class__.__name__}]"


# --- Main Walker ---

def walk_directory(start_path):
    """Walks the directory and prints the structure with summaries."""
    for root, dirs, files in os.walk(start_path, topdown=True):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * level
        
        # Print directory
        dir_name = os.path.basename(root)
        if level == 0:
            dir_name = os.path.basename(os.path.abspath(start_path)) # Show full name for root
        print(f"{indent}📂 {dir_name}/")

        file_indent = ' ' * 4 * (level + 1)
        for f in files:
            filepath = os.path.join(root, f)
            summary = None
            
            # --- File Type Logic ---
            if f.endswith('.py'):
                summary = get_python_docstring(filepath)
            elif f.endswith('.md'):
                summary = get_md_summary(filepath)
            elif f.endswith('.yml') or f.endswith('.yaml'):
                summary = get_yaml_description(filepath)
            
            # Print file and its summary
            if summary:
                print(f"{file_indent}📄 {f}  ->  ({summary})")
            else:
                print(f"{file_indent}📄 {f}")

# --- Execution ---

if __name__ == "__main__":
    # Use the directory provided as an argument, or default to the current directory
    if len(sys.argv) > 1:
        start_dir = sys.argv[1]
    else:
        start_dir = '.'

    if not os.path.isdir(start_dir):
        print(f"Error: Path '{start_dir}' is not a valid directory.")
        sys.exit(1)
        
    walk_directory(start_dir)