"""Tests for publish_causal_inference.py; run with python3 -m unittest from scripts/."""

import subprocess
import tempfile
import unittest
from pathlib import Path

import publish_causal_inference as pub


def touch(p: Path, text: str = "x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


class SelectNotesTest(unittest.TestCase):
    def test_takes_top_level_md_except_denylist_and_renames_main(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d)
            for name in ["causal_inference.notes.md", "p51.md", "gaps.md",
                         "causal_inference.pdf", "docs/nested.md"]:
                touch(src / name)
            got = pub.select_notes(src)
        self.assertEqual(got, {"causal_inference.notes.md": "elements-of-causal-inference.md",
                               "p51.md": "p51.md"})


class CopyNotesTest(unittest.TestCase):
    def test_copies_md_rewrites_pdf_links_copies_figures_and_no_pdf(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "src", Path(d) / "dst"
            touch(src / "p51.md", "see [[causal_inference.pdf#page=51]]")
            touch(src / "gaps.md")
            touch(src / "causal_inference.pdf")
            touch(src / "figures/a.png")
            written = pub.copy_notes(src, dst)
            self.assertEqual((dst / "p51.md").read_text(), "see *p. 51*")
            self.assertTrue((dst / "figures/a.png").exists())
            self.assertFalse((dst / "gaps.md").exists())
            self.assertEqual(list(dst.rglob("*.pdf")), [])
            self.assertEqual(written, ["p51.md"])


class GitPublishTest(unittest.TestCase):
    def _git(self, cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, check=True,
                              capture_output=True, text=True).stdout

    def test_commits_and_pushes_changes_then_noop_when_clean(self):
        with tempfile.TemporaryDirectory() as d:
            remote, repo = Path(d) / "remote.git", Path(d) / "repo"
            self._git(d, "init", "--bare", "-b", "main", str(remote))
            self._git(d, "clone", "-q", str(remote), str(repo))
            self._git(repo, "config", "user.email", "t@t")
            self._git(repo, "config", "user.name", "t")
            touch(repo / "README", "init")
            self._git(repo, "add", "."); self._git(repo, "commit", "-qm", "init")
            self._git(repo, "push", "-q", "-u", "origin", "main")

            touch(repo / "notes/Causal Inference/p51.md", "new")
            self.assertTrue(pub.git_publish(repo, "notes/Causal Inference"))
            self.assertIn("p51.md", self._git(d, "--git-dir", str(remote),
                                              "ls-tree", "-r", "--name-only", "main"))
            self.assertFalse(pub.git_publish(repo, "notes/Causal Inference"))


if __name__ == "__main__":
    unittest.main()


class WebBodyMacrosTest(unittest.TestCase):
    def test_expands_header_macros_and_drops_format_block(self):
        md = ("---\n"
              'title: "T"\n'
              "format:\n  pdf:\n    include-in-header:\n      text: |\n"
              "        \\DeclareMathOperator{\\Var}{Var}\n"
              "        \\newcommand{\\indep}{\\perp\\!\\!\\!\\perp}\n"
              "        \\newcommand{\\B}{\\mathcal{B}}\n"
              "---\n\n"
              "$X \\indep Y$, $\\Var(\\mu)$, $\\B$ and $\\Bigg($ stays.\n")
        self.assertEqual(pub.web_body(md),
                         '---\ntitle: "T"\n---\n\n'
                         "$X \\perp\\!\\!\\!\\perp Y$, $\\operatorname{Var}(\\mu)$, "
                         "$\\mathcal{B}$ and $\\Bigg($ stays.\n")
