"""Compile report/latex/main.tex and version the output as report_{N+1}.pdf.

Usage: uv run python report/build.py  (or plain `python report/build.py`)

Output: report/pdfs/report_N.pdf (auto-incremented) and report_latest.pdf.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LATEX_DIR = HERE / "latex"
PDFS_DIR = HERE / "pdfs"
MAIN_TEX = LATEX_DIR / "main.tex"
MAIN_PDF = LATEX_DIR / "main.pdf"

VERSION_RE = re.compile(r"^report_(\d+)\.pdf$")


def run(cmd: list[str], cwd: Path) -> int:
    print(f"$ {' '.join(cmd)}  (in {cwd})")
    return subprocess.run(cmd, cwd=cwd).returncode


def compile_latex() -> int:
    if shutil.which("latexmk"):
        return run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=LATEX_DIR,
        )
    print("warning: latexmk not found, falling back to pdflatex+biber", file=sys.stderr)
    for cmd in (
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["biber", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ):
        rc = run(cmd, cwd=LATEX_DIR)
        if rc != 0 and cmd[0] != "biber":
            return rc
    return 0


def next_version() -> int:
    existing = [
        int(m.group(1))
        for p in PDFS_DIR.iterdir()
        if p.is_file() and (m := VERSION_RE.match(p.name))
    ]
    return (max(existing) + 1) if existing else 1


def main() -> int:
    if not MAIN_TEX.exists():
        print(f"error: {MAIN_TEX} not found", file=sys.stderr)
        return 2
    PDFS_DIR.mkdir(parents=True, exist_ok=True)

    rc = compile_latex()
    if rc != 0 or not MAIN_PDF.exists():
        print("error: LaTeX compile failed", file=sys.stderr)
        return rc or 1

    n = next_version()
    versioned = PDFS_DIR / f"report_{n}.pdf"
    latest = PDFS_DIR / "report_latest.pdf"
    shutil.copy2(MAIN_PDF, versioned)
    shutil.copy2(MAIN_PDF, latest)
    print(f"wrote {versioned.relative_to(HERE.parent)}")
    print(f"wrote {latest.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
