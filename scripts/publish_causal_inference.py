#!/usr/bin/env python3
"""Copy Obsidian notes from ~/Documents/causal_inference into notes/Causal Inference/; --push commits and pushes."""

import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

SRC = Path.home() / "Documents/causal_inference"
REPO = Path(__file__).resolve().parent.parent
DST_REL = "notes/Causal Inference"

DENY = {"gaps.md"}
RENAME = {"causal_inference.notes.md": "elements-of-causal-inference.md"}


def select_notes(src: Path) -> dict[str, str]:
    """Top-level .md files minus DENY, mapped to destination names."""
    return {p.name: RENAME.get(p.name, p.name)
            for p in sorted(src.glob("*.md")) if p.name not in DENY}


def _split_frontmatter(md: str) -> tuple[str, str]:
    m = re.match(r"---\n(.*?)\n---\n", md, re.S)
    return (m.group(1), md[m.end():]) if m else ("", md)


def web_body(md: str) -> str:
    """Rewrite for the web: PDF links -> page numbers, expand header LaTeX macros, drop format: block."""
    md = re.sub(r"\[\[[^\]]+\.pdf#page=(\d+)\]\]", r"*p. \1*", md)
    fm, body = _split_frontmatter(md)
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", fm, re.M))
    macros |= {n: f"\\operatorname{{{v}}}" for n, v in
               re.findall(r"\\DeclareMathOperator\{\\(\w+)\}\{(.*)\}\s*$", fm, re.M)}
    for name, expansion in macros.items():
        body = re.sub(rf"\\{name}(?![A-Za-z])", lambda _: expansion, body)
    fm = re.sub(r"^format:\n(?:[ \t]+.*\n?)*", "", fm, flags=re.M).rstrip("\n")
    return f"---\n{fm}\n---\n{body}" if fm else body


def copy_notes(src: Path, dst: Path) -> list[str]:
    """Write selected notes and figures/ into dst; returns written note names."""
    dst.mkdir(parents=True, exist_ok=True)
    written = []
    for name, out in select_notes(src).items():
        (dst / out).write_text(web_body((src / name).read_text()))
        written.append(out)
    if (src / "figures").is_dir():
        shutil.copytree(src / "figures", dst / "figures", dirs_exist_ok=True)
    return written


def git_publish(repo: Path, path: str) -> bool:
    """Stage path, commit, push. Returns False when nothing changed."""
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True)
    git("add", "-A", path)
    if not git("status", "--porcelain", "--", path).stdout.strip():
        return False
    git("commit", "-qm", f"Publish causal inference notes {date.today():%Y-%m-%d}")
    git("push", "-q")
    return True


if __name__ == "__main__":
    for name in copy_notes(SRC, REPO / DST_REL):
        print("wrote", name)
    if "--push" in sys.argv:
        print("pushed" if git_publish(REPO, DST_REL) else "nothing to publish")
