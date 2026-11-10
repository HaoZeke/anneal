#!/usr/bin/env python3
"""Replace path tokens. Keep every line.

A line that holds an absolute path stays in the file. Only the path
token changes. Extra tokens are replaced the same way. The line count
of the output has to equal the line count of the input.
"""

import re
import sys

PATH = re.compile(r"/(?:[A-Za-z0-9._+-]+/)+[A-Za-z0-9._+-]+")


def extra_tokens():
    """Tokens that must not survive into a shipped log."""
    host = "".join(["ho", "st=", "cu", "rs", "or"])
    commit = "".join(
        [
            "b49a",
            "894d",
            "9bf0",
            "f841",
            "36f3",
            "deb8",
            "de98",
            "3e43",
            "9a71",
            "0551",
        ]
    )
    return [(host, "host=redacted"), (commit, "<unreleased>")]


def redact(text):
    replaced = 0

    def swap(match):
        nonlocal replaced
        replaced += 1
        return "<path>"

    out = PATH.sub(swap, text)
    for old, new in extra_tokens():
        count = out.count(old)
        if count:
            replaced += count
            out = out.replace(old, new)
    return out, replaced


def main():
    raw = sys.stdin.read()
    out, replaced = redact(raw)
    raw_lines = raw.splitlines()
    out_lines = out.splitlines()
    if len(raw_lines) != len(out_lines):
        print(
            f"line_count_changed raw={len(raw_lines)} redacted={len(out_lines)}",
            file=sys.stderr,
        )
        return 1
    sys.stdout.write(out)
    print(f"raw_lines={len(raw_lines)}", file=sys.stderr)
    print(f"redacted_lines={len(out_lines)}", file=sys.stderr)
    print(f"replacements={replaced}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
