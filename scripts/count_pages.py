#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# ///

from __future__ import annotations

import argparse
import re
from pathlib import Path

PAGE_RE = re.compile(
    r"pages\s*=\s*(?:\{([^}]*)\}|\"([^\"]*)\")",
    re.IGNORECASE,
)

TITLE_RE = re.compile(
    r"title\s*=\s*(?:\{([^}]*)\}|\"([^\"]*)\")",
    re.IGNORECASE,
)


def page_count(page_string: str) -> int | None:
    page_string = page_string.strip()

    # Accept both 123--130 and 123-130
    m = re.fullmatch(r"(\d+)\s*-{1,2}\s*(\d+)", page_string)
    if m:
        start = int(m.group(1))
        end = int(m.group(2))
        if end >= start:
            return end - start + 1
        return None

    # Single page
    if re.fullmatch(r"\d+", page_string):
        return 1

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count the total number of pages in a BibTeX file."
    )
    parser.add_argument("bibfile", type=Path)
    args = parser.parse_args()

    text = args.bibfile.read_text(encoding="utf-8")

    entries = text.split("@")[1:]

    total = 0
    counted = 0
    skipped = 0

    for entry in entries:
        pages_match = PAGE_RE.search(entry)
        if not pages_match:
            continue

        pages = pages_match.group(1) or pages_match.group(2)

        title_match = TITLE_RE.search(entry)
        title = (
            title_match.group(1) or title_match.group(2)
            if title_match
            else "(untitled)"
        )

        count = page_count(pages)

        if count is None:
            print(f"SKIP: {title[:60]} ({pages})")
            skipped += 1
            continue

        print(f"{count:4d} pages  {title[:60]}")
        total += count
        counted += 1

    print("\nSummary")
    print("-------")
    print(f"Entries counted : {counted}")
    print(f"Entries skipped : {skipped}")
    print(f"Total pages     : {total}")


if __name__ == "__main__":
    main()
