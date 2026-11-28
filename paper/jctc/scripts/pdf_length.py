# Copyright (c) anneal developers
"""Words and numerals of the rendered manuscript, as pdftotext reads them.

The abstract runs from the "Abstract" heading to the "Introduction" heading
and the body from "Introduction" to "Author contributions", so the body holds
the sections, tables, and captions and leaves out the back matter and the
references. Words are the whitespace tokens of poppler pdftotext with no
-layout flag, including a page number pdftotext prints on its own line.
A line that is only a section or subsection number, directly before that
heading, is not a word. A numeral is a run of digits, with any decimal
point or thousands comma inside it, so the 2 of GFN2 counts. A line that
is only a page number is not an abstract numeral.

    python3 paper/jctc/scripts/pdf_length.py --pdf /tmp/texb/manuscript.pdf --write
    python3 paper/jctc/scripts/pdf_length.py --pdf /tmp/texb/manuscript.pdf --check

--check fails when a count is outside its limit, and when the PDF page
count is not the pages= value in the receipt.

--write records the counts and the sha256 of the sources in
paper/jctc/logs/pdf-length.txt, which scripts/paper_audit.py reads.
abstract_words is the poppler pdftotext count of the abstract block
only: default extraction, not layout mode, from the Abstract heading
to the Introduction heading. Words are whitespace tokens, including a
page number on its own line. A section number on its own line before
that heading is not a word. A page number on its own line is not an
abstract numeral. The receipt names that method.
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
MANUSCRIPT = ROOT / "paper/jctc/manuscript.tex"
SOURCES = ("paper/jctc/manuscript.tex", "paper/jctc/macros.tex", "paper/jctc/labels-si.tex")
ABSTRACT_WORDS = 200
ABSTRACT_NUMERALS = 6
BODY_WORDS = (5000, 7000)
NUMERAL = re.compile(r"\d+(?:[.,]\d+)*")


def pdf_text(pdf: Path) -> str:
    return subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout


def numbered_headings(tex: str) -> list[tuple[str, str]]:
    """Section and subsection numbers in the order achemso assigns them."""
    text = re.sub(r"(?m)%.*", "", tex)
    found = []
    sec = sub = 0
    for match in re.finditer(r"\\(section|subsection)(\*)?\{", text):
        kind, star, start = match.group(1), match.group(2), match.end()
        depth = 1
        chars = []
        index = start
        while index < len(text) and depth:
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            chars.append(char)
            index += 1
        title = re.sub(r"\\[a-zA-Z]+|[{}]", "", "".join(chars))
        title = re.sub(r"\s+", " ", title).strip()
        if star:
            continue
        if kind == "section":
            sec += 1
            sub = 0
            number = str(sec)
        else:
            sub += 1
            number = f"{sec}.{sub}"
        found.append((number, title))
    return found


def without_section_numbers(chunk: str, headings: list[tuple[str, str]]) -> str:
    """Drop a line that is only the number of the heading on the next line."""
    lines = chunk.split("\n")
    drop = set()
    for index, line in enumerate(lines):
        number = line.strip()
        if not re.fullmatch(r"\d+(?:\.\d+)?", number):
            continue
        nxt = index + 1
        while nxt < len(lines) and not lines[nxt].strip():
            nxt += 1
        if nxt >= len(lines):
            continue
        title = re.sub(r"\s+", " ", lines[nxt].strip())
        for heading_number, heading in headings:
            if number == heading_number and (title == heading or title.startswith(heading)):
                drop.add(index)
    return "\n".join(line for index, line in enumerate(lines) if index not in drop)


def without_digit_lines(chunk: str) -> str:
    """Drop a line that is only a page number, so it is not an abstract numeral."""
    return "\n".join(line for line in chunk.split("\n") if not re.fullmatch(r"\s*\d+\s*", line))


def between(text: str, start: str, end: str) -> str:
    begin = text.index(f"\n{start}\n") + len(start) + 2
    return text[begin:text.index(f"\n{end}\n", begin)]


def words(chunk: str) -> int:
    return len(chunk.split())


def numerals(chunk: str) -> int:
    return len(NUMERAL.findall(chunk))


def measure(text: str, tex: str | None = None) -> dict[str, int]:
    text = text.replace("\f", "\n")
    if tex is None and MANUSCRIPT.is_file():
        tex = MANUSCRIPT.read_text()
    headings = numbered_headings(tex or "")
    abstract = between(text, "Abstract", "Introduction")
    body = without_section_numbers(between(text, "Introduction", "Author contributions"), headings)
    return {
        "abstract_words": words(abstract),
        "abstract_numerals": numerals(without_digit_lines(abstract)),
        "body_words": words(body),
    }


def source_digests(root: Path = ROOT) -> dict[str, str]:
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in SOURCES}


def page_problem(pages: int, recorded: dict[str, int]) -> list[str]:
    """The rendered page count must be the receipt's pages= value."""
    if recorded.get("pages") == pages:
        return []
    shown = recorded["pages"] if "pages" in recorded else "absent"
    return [f"pages {pages}, receipt has {shown}"]


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
    lines = [
        "# abstract_words: poppler pdftotext default, no -layout, whitespace tokens",
        "# a section number alone before its heading is not a word; a page number is not a numeral",
    ]
    lines += [f"{key}={value}" for key, value in counts.items()] + [f"pages={pages}"]
    lines += [f"sha256 {digest}  {path}" for path, digest in digests.items()]
    return "\n".join(lines) + "\n"


def read_receipt(path: Path = RECEIPT) -> tuple[dict[str, int], dict[str, str]]:
    counts: dict[str, int] = {}
    digests: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
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
    if args.check and RECEIPT.is_file():
        recorded, _digests = read_receipt()
        found.extend(page_problem(pages, recorded))
    for problem in found:
        print(f"pdf_length: {problem}", file=sys.stderr)
    return 1 if args.check and found else 0


if __name__ == "__main__":
    raise SystemExit(main())
