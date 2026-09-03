#!/usr/bin/env python3
"""Copy the Obsidian notes from ~/Documents/causal_inference into notes/Causal Inference/."""

import re
import shutil
from pathlib import Path

SRC = Path.home() / "Documents/causal_inference"
DST = Path(__file__).resolve().parent.parent / "notes/Causal Inference"

MD_FILES = {
    "causal_inference.notes.md": "elements-of-causal-inference.md",
    "p36.md": "p36.md",
    "p415.md": "p415.md",
    "p416.md": "p416.md",
    "p51.md": "p51.md",
    "deconv_helpers.md": "deconv_helpers.md",
}


def web_body(md: str) -> str:
    """PDF page links have no target on the web; print the page number instead."""
    return re.sub(r"\[\[[^\]]+\.pdf#page=(\d+)\]\]", r"*p. \1*", md)


if __name__ == "__main__":
    DST.mkdir(exist_ok=True)
    for src, dst in MD_FILES.items():
        (DST / dst).write_text(web_body((SRC / src).read_text()))
        print("wrote", dst)
    if (SRC / "figures").is_dir():
        shutil.copytree(SRC / "figures", DST / "figures", dirs_exist_ok=True)
        print("copied figures/")
