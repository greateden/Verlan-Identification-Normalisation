## 🐍 Python Script（自動生成目錄樹）
# update_tree.py
# Python 3.10+
# Dependencies: none (built-in only)

import os

def generate_tree(start_path, prefix=""):
    tree_str = ""
    files = sorted(os.listdir(start_path))
    entries = [f for f in files if not f.startswith('.')]  # skip hidden files

    for i, entry in enumerate(entries):
        path = os.path.join(start_path, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        tree_str += f"{prefix}{connector}{entry}\n"
        if os.path.isdir(path):
            extension = "    " if i == len(entries) - 1 else "│   "
            tree_str += generate_tree(path, prefix + extension)
    return tree_str

if __name__ == "__main__":
    project_root = ".."  # current directory
    print("project-root/")
    print(generate_tree(project_root))