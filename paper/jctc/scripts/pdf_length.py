# Copyright (c) anneal developers
"""Words and numerals of the rendered manuscript, as pdftotext reads them.

The abstract runs from the "Abstract" heading to the "Introduction" heading
and the body from "Introduction" to "Author contributions", so the body holds
the sections, tables, and captions and leaves out the back matter and the
references. Page numbers are not words. A numeral is
a run of digits, with any decimal point or thousands comma inside it, so the
2 of GFN2 counts.

    python3 paper/jctc/scripts/pdf_length.py --pdf /tmp/texb/manuscript.pdf --write
    python3 paper/jctc/scripts/pdf_length.py --pdf /tmp/texb/manuscript.pdf --check

--write records the counts and the sha256 of the sources in
paper/jctc/logs/pdf-length.txt, which scripts/paper_audit.py reads.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "paper/jctc/logs/pdf-length.txt"
SOURCES = ("paper/jctc/manuscript.tex", "paper/jctc/macros.tex", "paper/jctc/labels-si.tex")
ABSTRACT_WORDS = 200
ABSTRACT_NUMERALS = 6
BODY_WORDS = (5000, 7000)
NUMERAL = re.compile(r"\d+(?:[.,]\d+)*")


def pdf_text(pdf: Path) -> str:
    return subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout


def without_page_numbers(text: str) -> str:
    """Drop the page number pdftotext prints as the last line of each page."""
    pages = []
    for page in text.split("\f"):
        lines = page.rstrip("\n").split("\n")
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and re.fullmatch(r"\s*\d+\s*", lines[-1]):
            lines.pop()
        pages.append("\n".join(lines))
    return "\n".join(pages)


def between(text: str, start: str, end: str) -> str:
    begin = text.index(f"\n{start}\n") + len(start) + 2
    return text[begin:text.index(f"\n{end}\n", begin)]


def words(chunk: str) -> int:
    return len(chunk.split())


def numerals(chunk: str) -> int:
    return len(NUMERAL.findall(chunk))


def measure(text: str) -> dict[str, int]:
    text = without_page_numbers(text)
    abstract = re.sub(r"\n\s*\d+\s*$", "", between(text, "Abstract", "Introduction"))
    body = between(text, "Introduction", "Author contributions")
    return {"abstract_words": words(abstract), "abstract_numerals": numerals(abstract), "body_words": words(body)}


def source_digests(root: Path = ROOT) -> dict[str, str]:
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in SOURCES}


def problems(counts: dict[str, int]) -> list[str]:
    found = []
    if counts["abstract_words"] > ABSTRACT_WORDS:
        found.append(f"abstract has {counts['abstract_words']} words, more than {ABSTRACT_WORDS}")
    if counts["abstract_numerals"] > ABSTRACT_NUMERALS:
        found.append(f"abstract has {counts['abstract_numerals']} numerals, more than {ABSTRACT_NUMERALS}")
    low, high = BODY_WORDS
    if not low <= counts["body_words"] <= high:
        found.append(f"body has {counts['body_words']} words, outside {low}-{high}")
    return found


def receipt_text(counts: dict[str, int], pages: int, digests: dict[str, str]) -> str:
    lines = [f"{key}={value}" for key, value in counts.items()] + [f"pages={pages}"]
    lines += [f"sha256 {digest}  {path}" for path, digest in digests.items()]
    return "\n".join(lines) + "\n"


def read_receipt(path: Path = RECEIPT) -> tuple[dict[str, int], dict[str, str]]:
    counts: dict[str, int] = {}
    digests: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.startswith("sha256 "):
            digest, source = line.removeprefix("sha256 ").split("  ", 1)
            digests[source] = digest
        else:
            key, _, value = line.partition("=")
            counts[key] = int(value)
    return counts, digests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help=f"write {RECEIPT.relative_to(ROOT)}")
    parser.add_argument("--check", action="store_true", help="fail on a count outside its limit")
    args = parser.parse_args()
    text = pdf_text(args.pdf)
    counts = measure(text)
    pages = text.count("\f")
    for key, value in counts.items():
        print(f"{key} {value}")
    print(f"pages {pages}")
    if args.write:
        RECEIPT.write_text(receipt_text(counts, pages, source_digests()))
    found = problems(counts)
    for problem in found:
        print(f"pdf_length: {problem}", file=sys.stderr)
    return 1 if args.check and found else 0


if __name__ == "__main__":
    raise SystemExit(main())
