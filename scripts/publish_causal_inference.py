#!/usr/bin/env python3
"""Publish selected files from ~/Documents/causal_inference into notes/Causal Inference/.
tex files become standalone md pages; the notes qmd gets web frontmatter and input->links."""

import re
import subprocess
from pathlib import Path

SRC = Path.home() / "Documents/causal_inference"
DST = Path(__file__).resolve().parent.parent / "notes/Causal Inference"

TEX_FILES = ["p36.tex", "p415.tex", "p416.tex", "p51.tex", "deconv_helpers.tex"]
NOTES_QMD = "causal_inference.notes.qmd"

WEB_FRONTMATTER = """---
title: "Notes: Elements of Causal Inference (Peters, Janzing, Schölkopf)"
from: markdown+tex_math_single_backslash
---
"""


def tex_title(tex: Path) -> str:
    m = re.search(r"\\textbf\{([^}]+)\}", tex.read_text())
    return m.group(1).rstrip(". ") if m else tex.stem


def normalize_math(md: str) -> str:
    """Put $$ display math on its own lines with blank lines around; Quartz's
    parser mis-splits fences that share a line with text."""
    parts = md.split("$$")
    if len(parts) % 2 == 0:
        return md
    for i in range(1, len(parts), 2):
        parts[i] = "\n" + parts[i].strip() + "\n"
    for i in range(0, len(parts), 2):
        p = parts[i]
        if i > 0:
            p = p.lstrip(" \t")
            p = p if p.startswith("\n\n") else ("\n" + p if p.startswith("\n") else "\n\n" + p)
        if i < len(parts) - 1:
            p = p.rstrip(" \t")
            p = p if p.endswith("\n\n") else (p + "\n" if p.endswith("\n") else p + "\n\n")
        parts[i] = p
    return "$$".join(parts)


def convert_tex(tex: Path) -> None:
    md = subprocess.run(
        ["quarto", "pandoc", "-f", "latex", "-t", "gfm+tex_math_dollars-tex_math_gfm", str(tex)],
        check=True, capture_output=True, text=True,
    ).stdout
    out = DST / f"{tex.stem}.md"
    out.write_text(f'---\ntitle: "{tex_title(tex)}"\n---\n\n{normalize_math(md)}')
    print(f"wrote {out.name} ({tex_title(tex)})")


def convert_notes() -> None:
    body = (SRC / NOTES_QMD).read_text()
    body = re.sub(r"^---\n.*?\n---\n", "", body, count=1, flags=re.S)
    def input_to_link(m: re.Match) -> str:
        stem = Path(m.group(1)).stem
        return f"[{tex_title(SRC / m.group(1))}]({stem})"
    body = re.sub(r"\\input\{([^}]+)\}", input_to_link, body)
    out = DST / "elements-of-causal-inference.qmd"
    out.write_text(WEB_FRONTMATTER + body)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    DST.mkdir(exist_ok=True)
    for name in TEX_FILES:
        convert_tex(SRC / name)
    (DST / "plots.py").write_bytes((SRC / "plots.py").read_bytes())
    print("copied plots.py")
    convert_notes()
    print("now: source .venv/bin/activate && quarto render notes")
