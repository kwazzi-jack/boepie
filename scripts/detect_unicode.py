# /// script
# requires-python = ">=3.11"
# dependencies = ["rich"]
# ///
import sys
import unicodedata
from rich.console import Console
from rich.text import Text

console = Console()

for lineno, line in enumerate(sys.stdin, 1):
    line = line.rstrip("\n")
    if all(ord(c) < 128 for c in line):
        continue

    text = Text()
    text.append(f"{lineno}: ", style="dim")
    names = []
    for c in line:
        if ord(c) < 128:
            text.append(c)
        else:
            text.append(c, style="bold white on red")
            try:
                name = unicodedata.name(c)
            except ValueError:
                name = "UNNAMED"
            names.append(f"U+{ord(c):04X} {name}")

    console.print(text)
    if names:
        console.print("    " + ", ".join(names), style="yellow")
