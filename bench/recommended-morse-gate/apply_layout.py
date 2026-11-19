#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Change only the size and padding of cluster Config.

Three layouts, applied to a clean worktree of arm B:

- end64: #[repr(C)] and 64 unread bytes at the end
- start256: #[repr(C)] and 256 unread bytes as the first field
- align1024: #[repr(C, align(64))] and 1024 unread bytes at the end

The bytes are zero, skipped by serde, and passed through std::hint::black_box
so the field remains in the value. No other field initializer changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAYOUTS = {
    "end64": {"n": 64, "repr": "#[repr(C)]\n", "first": False},
    "start256": {"n": 256, "repr": "#[repr(C)]\n", "first": True},
    "align1024": {"n": 1024, "repr": "#[repr(C, align(64))]\n", "first": False},
}

STRUCT_ANCHOR = (
    "/// Driver settings.\n"
    "#[derive(Debug, Clone, serde::Serialize)]\n"
    "pub struct Config {\n"
)
FIELD_ANCHOR = "    pub branch_on_stall: bool,\n}\n"
OPEN_ANCHOR = (
    "        assert!(\n"
    '            energy_scale.is_finite() && energy_scale > 0.0,\n'
    '            "energy_scale must be finite and positive"\n'
    "        );\n"
    "        Self {\n"
)
CLOSE_ANCHOR = (
    "            branch_on_stall: false,\n"
    "        }\n"
    "    }\n"
    "\n"
    "    /// Basin-hopping settings for a rigid TIP4P water cluster.\n"
)


def pad_field(n: int) -> str:
    return (
        "    /// Unread bytes that change this value's size and alignment.\n"
        "    #[serde(skip)]\n"
        f"    pub layout_pad: [u8; {n}],\n"
    )


def apply_layout(text: str, kind: str) -> str:
    spec = LAYOUTS[kind]
    n = spec["n"]
    if "layout_pad" in text:
        raise SystemExit("layout_pad is already present")
    for anchor in (STRUCT_ANCHOR, FIELD_ANCHOR, OPEN_ANCHOR, CLOSE_ANCHOR):
        if text.count(anchor) != 1:
            raise SystemExit(f"anchor count {text.count(anchor)} for a required site")
    struct = (
        "/// Driver settings.\n"
        "#[derive(Debug, Clone, serde::Serialize)]\n"
        f"{spec['repr']}"
        "pub struct Config {\n"
    )
    if spec["first"]:
        struct += pad_field(n)
    text = text.replace(STRUCT_ANCHOR, struct, 1)
    if not spec["first"]:
        text = text.replace(FIELD_ANCHOR, "    pub branch_on_stall: bool,\n" + pad_field(n) + "}\n", 1)
    text = text.replace(OPEN_ANCHOR, OPEN_ANCHOR.replace("        Self {\n", "        let built = Self {\n"), 1)
    close = (
        "            branch_on_stall: false,\n"
        f"            layout_pad: [0; {n}],\n"
        "        };\n"
        "        let _kept = std::hint::black_box(built.layout_pad);\n"
        "        built\n"
        "    }\n"
        "\n"
        "    /// Basin-hopping settings for a rigid TIP4P water cluster.\n"
    )
    text = text.replace(CLOSE_ANCHOR, close, 1)
    if (
        text.count("pub layout_pad:") != 1
        or text.count("layout_pad: [0;") != 1
        or text.count("built.layout_pad") != 1
    ):
        raise SystemExit("layout pad was not inserted in the struct, the literal, and the barrier")
    return text


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=sorted(LAYOUTS))
    parser.add_argument("worktree", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    path = args.worktree / "src/methods/cluster_hopping/config.rs"
    original = path.read_text()
    updated = apply_layout(original, args.kind)
    if args.check:
        apply_layout(original, "end64")
        apply_layout(original, "start256")
        apply_layout(original, "align1024")
        return 0
    path.write_text(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
