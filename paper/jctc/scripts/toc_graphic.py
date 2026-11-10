#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the TOC graphic slot at the JCTC size, 3.25 in by 1.75 in.

The drawing is three labeled boxes: potential, hop, and bank.
No campaign count is printed. From the repository root:

    python3 paper/jctc/scripts/toc_graphic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

WIDTH = 3.25 * 72
HEIGHT = 1.75 * 72


def stream() -> bytes:
    """PDF content stream: a frame and three boxes."""
    boxes = [
        (12, 36, 66, 78, "potential"),
        (90, 36, 144, 78, "hop"),
        (168, 36, 222, 78, "bank"),
    ]
    parts = ["0.15 w", "12 18 210 96 re S"]
    for x0, y0, x1, y1, _label in boxes:
        parts.append(f"{x0} {y0} {x1 - x0} {y1 - y0} re S")
    parts.append("BT /F1 9 Tf")
    for x0, y0, x1, _y1, label in boxes:
        parts.append(f"1 0 0 1 {x0 + 4} {y0 + 16} Tm ({label}) Tj")
    parts.append("ET")
    body = "\n".join(parts).encode("ascii")
    return body


def pdf(body: bytes) -> bytes:
    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>"
    )
    media = f"[0 0 {WIDTH:.2f} {HEIGHT:.2f}]".encode("ascii")
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox "
        + media
        + b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    )
    objects.append(
        b"<< /Length "
        + str(len(body)).encode("ascii")
        + b" >>\nstream\n"
        + body
        + b"\nendstream"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        dest = Path(args[0])
    else:
        dest = Path(__file__).resolve().parents[1] / "figs" / "toc-graphic.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(pdf(stream()))


if __name__ == "__main__":
    main()
