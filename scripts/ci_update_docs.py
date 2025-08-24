#!/usr/bin/env python3
# scripts/ci_update_docs.py
# Python 3.11+ | no external deps
# Purpose: Update README blocks for (1) repo tree and (2) NZT deadline countdown.
# Usage:
#   python scripts/ci_update_docs.py --root docs/readme.md --base-dir .
#
# Notes:
# - Uses Python's stdlib zoneinfo for Pacific/Auckland timezone.
# - Commits happen in the workflow only if file content changes.

import os
import re
import argparse
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo  # Python >=3.9

# ---------- Config ----------
IGNORE_DIRS = {".git", ".github", "__pycache__", ".mypy_cache", ".pytest_cache",
               ".DS_Store", ".venv", "venv", ".idea", ".vscode"}
IGNORE_FILES = {".DS_Store"}

TREE_START = "<!-- TREE:START -->"
TREE_END   = "<!-- TREE:END -->"

DUE_START = "<!-- DUE:START -->"
DUE_END   = "<!-- DUE:END -->"

# Project deadline: 2025-10-17 24:00 NZT == 2025-10-18 00:00 in Pacific/Auckland
NZ_TZ = ZoneInfo("Pacific/Auckland")
DUE_NZ = datetime(2025, 10, 18, 0, 0, 0, tzinfo=NZ_TZ)  # 00:00 on Oct 18 local
# --------------------------------------

def generate_tree(start_path: str, prefix: str = "", is_root: bool = True) -> str:
    """
    Recursively walk the repo and build an ASCII tree.
    Skips hidden/ignored paths. Only stdlib.
    """
    entries = []
    for name in sorted(os.listdir(start_path)):
        if name in IGNORE_FILES:
            continue
        full = os.path.join(start_path, name)
        base = os.path.basename(full)
        if os.path.isdir(full):
            if base in IGNORE_DIRS or base.startswith('.'):
                continue
            entries.append((base + "/", full, True))
        else:
            if base.startswith('.'):
                continue
            entries.append((base, full, False))

    lines = []
    if is_root:
        lines.append("project-root/")
    for idx, (disp, full, is_dir) in enumerate(entries):
        connector = "└── " if idx == len(entries) - 1 else "├── "
        lines.append(f"{prefix}{connector}{disp}")
        if is_dir:
            continuation = "    " if idx == len(entries) - 1 else "│   "
            subtree = generate_tree(full, prefix + continuation, is_root=False)
            if subtree:
                lines.extend(subtree.splitlines())
    return "\n".join(lines)

def fmt_timedelta(delta: timedelta) -> str:
    """
    Render a timedelta into 'X days, HH hours, MM minutes'.
    Floors toward zero. Handles negative (deadline passed).
    """
    total_seconds = int(delta.total_seconds())
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    return f"{sign}{days} days, {hours:02d} hours, {minutes:02d} minutes"

def build_due_block() -> str:
    """
    Compute time remaining to the NZ deadline relative to current UTC.
    Shows both NZT and UTC instants for clarity.
    """
    now_utc = datetime.now(timezone.utc)
    due_utc = DUE_NZ.astimezone(timezone.utc)
    remaining = due_utc - now_utc

    status = "⏳ Time remaining" if remaining.total_seconds() > 0 else "✅ Deadline passed"
    when_nz = DUE_NZ.strftime("%Y-%m-%d %H:%M %Z")
    when_utc = due_utc.strftime("%Y-%m-%d %H:%M %Z")
    left = fmt_timedelta(remaining)

    # Fixed fenced code block with language `text`
    block = (
        f"{DUE_START}\n"
        "```text\n"
        f"{status}: {left}\n"
        f"Deadline (NZT): {when_nz}\n"
        f"Deadline (UTC): {when_utc}\n"
        "```\n"
        f"{DUE_END}"
    )
    return block

def replace_block(full_text: str, start_tag: str, end_tag: str, new_payload: str) -> str:
    """
    Replace any content between start_tag and end_tag (inclusive) with new_payload.
    If not found, append a new section at the end.
    """
    pattern = re.compile(rf"{re.escape(start_tag)}.*?{re.escape(end_tag)}",
                         flags=re.DOTALL | re.IGNORECASE)
    if pattern.search(full_text):
        return pattern.sub(new_payload, full_text)
    else:
        # Append a section if markers missing; choose sensible headings.
        heading = "## ⏳ Submission Countdown (NZT)\n\n" if start_tag == DUE_START else "## 📂 Repository Structure\n\n"
        return full_text.strip() + "\n\n" + heading + new_payload + "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="docs/readme.md", help="README path to update")
    parser.add_argument("--base-dir", default=".", help="Base directory to scan for tree")
    args = parser.parse_args()

    # Read or create README
    if os.path.exists(args.root):
        with open(args.root, "r", encoding="utf-8") as f:
            readme = f.read()
    else:
        os.makedirs(os.path.dirname(args.root), exist_ok=True)
        readme = "# Documentation\n"

    # Build tree block
    tree = generate_tree(args.base_dir)
    tree_block = (
        f"{TREE_START}\n```text\n{tree}\n```\n{TREE_END}"
    )

    # Build due block
    due_block = build_due_block()

    # Replace both blocks
    out = replace_block(readme, DUE_START, DUE_END, due_block)
    out = replace_block(out, TREE_START, TREE_END, tree_block)

    if out != readme:
        with open(args.root, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Updated {args.root}")
    else:
        print("No changes detected.")

if __name__ == "__main__":
    main()