# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click>=8.4.2",
#     "rich>=15.0.0",
# ]
# ///

from __future__ import annotations

from pathlib import Path
import re
import sys

import click
from rich.console import Console

console = Console()

_HEADER_PATTERN = re.compile(r"^#\s+([\w./-]+\.py)\s*$")


def is_text_file(path: Path, sample_size: int = 8192) -> bool:
    """Check if the file is a text file. Assumes the path is a file."""

    with path.open("rb") as file:
        chunk = file.read(sample_size)

    # Check for null byte
    if b"\x00" in chunk:
        return False  # -> most likely a binary file

    # Try decode the text as UTF-8
    try:
        chunk.decode("utf-8")
        return True
    # Could not decode as UTF-8
    except UnicodeDecodeError:
        return False


def find_repo_root(start: Path) -> Path:
    """Walks upward from `start` to the nearest ancestor with a pyproject.toml."""

    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    raise click.ClickException(f"Could not find a pyproject.toml above {start}")


def package_relative_path(path: Path, repo_root: Path) -> str:
    """`path` relative to `repo_root`, with a leading `src/` stripped so
    package files read as their import path, e.g. `boepie/cli.py`."""

    parts = path.resolve().relative_to(repo_root).parts
    if parts and parts[0] == "src":
        parts = parts[1:]
    return "/".join(parts)


def existing_header(first_line: str) -> str | None:
    """The path recorded in `first_line` if it looks like a header comment
    this script writes, else None."""

    match = _HEADER_PATTERN.match(first_line)
    return match.group(1) if match else None


def apply_header(path: Path, repo_root: Path, *, force: bool) -> bool:
    """Ensures `path`'s first line is `# <package_relative_path>`, replacing
    a stale header or inserting a new one as needed. Returns True if the file
    was written."""

    relative_path = package_relative_path(path, repo_root)
    correct_header = f"# {relative_path}"

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    first_line = lines[0] if lines else ""
    current_header = existing_header(first_line)

    if current_header == relative_path and not force:
        return False

    remaining_lines = lines[1:] if current_header is not None else lines
    path.write_text(f"{correct_header}\n{''.join(remaining_lines)}", encoding="utf-8")
    return True


@click.command()
@click.argument("file_or_dir_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--force",
    is_flag=True,
    help="Rewrite the header comment even on files that already have the correct one.",
)
def format_src(file_or_dir_path: Path, force: bool) -> None:
    """Formats a file or recursively formats files inside a directory using
    a few formatting rules."""

    if file_or_dir_path.is_file():
        paths = (file_or_dir_path,)
    else:
        paths = (path for path in file_or_dir_path.rglob("*") if path.is_file())

    python_paths = [path for path in paths if path.suffix == ".py" and is_text_file(path)]

    if len(python_paths) == 0:
        console.print("> No text-files to format -- nothing to do.")
        sys.exit(0)

    console.print(f"> Formatting {len(python_paths)} files")

    repo_root = find_repo_root(file_or_dir_path)
    updated_count = sum(apply_header(path, repo_root, force=force) for path in python_paths)

    console.print(f"> Updated header comment in {updated_count} of {len(python_paths)} files")


if __name__ == "__main__":
    format_src()
