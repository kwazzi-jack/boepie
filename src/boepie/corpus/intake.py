# src/boepie/corpus/intake.py
"""Turns an identifier the user hands `boepie corpus add` into a document.

One dispatch covers every collection: a local file is classified by suffix
and converted according to its format, an http(s) URL is fetched and its HTML
converted, and anything else is passed back for a collection-specific
resolver (an arXiv id, a DOI, a `.bib` file) to make sense of. What differs
per collection is only the frontmatter block written on top and which
resolvers run first - see `boepie.corpus.schema`.

Text-shaped formats (markdown, plain text, source code) are read verbatim: a
`.py` file's own bytes are already the best representation of it, and running
it through any converter would only lose information. Binary document
formats (PDF, DOCX, PPTX, XLSX) go to MinerU, which is an opt-in extra rather
than a hard dependency - it pulls model weights and wants a GPU to be quick,
which is too much to impose on someone who only ever adds Markdown notes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from boepie.corpus.schema import ConversionVia, SourceFormat

_USER_AGENT = "boepie-corpus-add (+https://github.com/kwazzi-jack/boepie)"
_REQUEST_TIMEOUT_SECONDS = 30

# Suffix -> format. Anything not listed and not obviously binary is treated as
# code, on the reasoning that an unrecognised text file is far more often a
# config or source file than something boepie should refuse.
_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".mdown"})
_TEXT_SUFFIXES = frozenset({".txt", ".text", ".rst", ".org", ".log"})
_HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})
_MINERU_SUFFIXES: dict[str, SourceFormat] = {
    ".pdf": "pdf", ".docx": "docx", ".pptx": "pptx", ".xlsx": "xlsx",
}
_CODE_SUFFIXES = frozenset({
    ".py", ".pyi", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".c", ".h", ".cc",
    ".cpp", ".hpp", ".go", ".rs", ".java", ".kt", ".rb", ".sh", ".bash",
    ".zsh", ".fish", ".ps1", ".sql", ".r", ".jl", ".m", ".f90", ".f", ".pro",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".xml", ".csv", ".tsv", ".tex", ".bib", ".lua", ".pl", ".php", ".swift",
    ".scala", ".hs", ".ex", ".exs", ".vim", ".dockerfile", ".makefile",
})

# Tried in order when a text file is not valid UTF-8. Latin-1 never fails, so
# it terminates the list and guarantees a text file is always readable rather
# than crashing the whole batch on one stray byte.
_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

# Fenced-block languages worth labelling, so a code note's fence carries the
# highlighting hint a reader (or a model) expects.
_FENCE_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".sh": "bash",
    ".bash": "bash", ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".toml": "toml", ".sql": "sql", ".c": "c", ".cpp": "cpp", ".go": "go",
    ".rs": "rust", ".java": "java", ".rb": "ruby", ".r": "r", ".jl": "julia",
}


class IntakeError(Exception):
    """A user-facing intake failure: an unreadable file, a missing converter,
    an identifier nothing could resolve. Carries a message meant to be shown
    as-is, so the CLI never has to translate an exception into advice."""


@dataclass(frozen=True)
class Converted:
    """One source turned into Markdown, with everything the `source` block needs."""

    markdown: str
    via: ConversionVia
    format: SourceFormat
    origin: str
    sha256: str | None = None
    # The source bytes, retained only when the caller asked to keep them.
    original_bytes: bytes | None = None
    original_name: str | None = None
    # Title suggested by the source itself (a leading H1, a page's <title>);
    # the caller falls back to this when no --title was given.
    suggested_title: str | None = None


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def looks_like_url(identifier: str) -> bool:
    return urlparse(identifier).scheme in ("http", "https")


def detect_format(path: Path) -> SourceFormat:
    """Classify a local file by suffix.

    Suffix rather than content sniffing: the formats that matter here are
    already unambiguous by extension, and a wrong guess on a binary file is
    caught by the converter anyway.
    """
    suffix = path.suffix.lower()
    if suffix in _MINERU_SUFFIXES:
        return _MINERU_SUFFIXES[suffix]
    if suffix in _MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in _HTML_SUFFIXES:
        return "html"
    if suffix in _TEXT_SUFFIXES:
        return "text"
    if suffix in _CODE_SUFFIXES or not suffix:
        return "code"
    return "code"


def read_text_file(path: Path) -> str:
    """Read a text file, trying progressively more forgiving encodings.

    The old notes intake called `read_text(encoding="utf-8")` directly, so a
    PDF (or any non-UTF-8 file) surfaced a raw `UnicodeDecodeError` as the
    CLI's error message. Binary formats never reach here now, but a text file
    written on a Windows box still might not be UTF-8.
    """
    data = path.read_bytes()
    for encoding in _TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IntakeError(f"could not decode '{path}' as text in any known encoding.")


# Inline markdown a heading may carry that a title should not. Converters
# vary in how much emphasis they preserve - MinerU renders a bold DOCX
# heading as `# **Title**` - and the markers would otherwise reach the title
# field, and from there every search hit, `corpus tree` line, and filename.
_MARKDOWN_INLINE = (
    re.compile(r"\[([^\]]+)\]\([^)]*\)"),  # [text](url)
    re.compile(r"\*\*(.+?)\*\*"),
    re.compile(r"__(.+?)__"),
    re.compile(r"\*(.+?)\*"),
    re.compile(r"`(.+?)`"),
)


def strip_markdown_inline(text: str) -> str:
    """Reduce one line of markdown to its plain text.

    Deliberately narrow: it unwraps emphasis and link syntax and nothing
    else, so an identifier like `snake_case_name` (whose underscores are not
    a matched pair around the whole word) survives intact.
    """
    cleaned = text
    for pattern in _MARKDOWN_INLINE:
        cleaned = pattern.sub(r"\1", cleaned)
    # Unmatched markers (a heading of literally `****`) survive the patterns
    # above, since each needs a pair with content between. Trimming them from
    # the ends leaves nothing, which the caller reads as "no usable title".
    return cleaned.strip(" *_`\t")


def title_from_markdown(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = strip_markdown_inline(stripped[2:])
            if title:
                return title
    return fallback


# ---------------------------------------------------------------------------
# MinerU: the opt-in binary-document converter
# ---------------------------------------------------------------------------
#
# Driven through its CLI rather than its Python API. MinerU imports torch and
# its model stack at import time, which would make every `boepie` invocation
# pay for a dependency almost no command uses; a subprocess also keeps a model
# crash from taking the CLI down with it.

_MINERU_MISSING = (
    "MinerU is required to convert {format} files and is not installed.\n"
    "Install it with:  uv sync --extra mineru\n"
    "Or convert the file yourself and add the resulting Markdown instead."
)


def mineru_available() -> bool:
    return shutil.which("mineru") is not None


# Settings whose value is "auto" mean "let MinerU decide", which MinerU
# expects to be expressed by the variable being *absent*. Passing the literal
# string through is an error it rejects outright:
#   MINERU_MODEL_SOURCE=auto is not supported. Unset MINERU_MODEL_SOURCE to
#   use auto detection once, or set it to huggingface/modelscope/local.
# The variable is actively removed rather than merely not set, so an "auto"
# already exported in the user's shell cannot reach MinerU either.
_MINERU_AUTO = "auto"
_MINERU_ENV_VARS = {
    "device_mode": "MINERU_DEVICE_MODE",
    "model_source": "MINERU_MODEL_SOURCE",
}


def _mineru_environment(*, device_mode: str, model_source: str) -> dict[str, str]:
    """The child environment for one MinerU run."""
    environment = dict(os.environ)
    for setting, variable in _MINERU_ENV_VARS.items():
        value = {"device_mode": device_mode, "model_source": model_source}[setting]
        if value == _MINERU_AUTO:
            environment.pop(variable, None)
        else:
            environment[variable] = value
    return environment


def _mineru_failure_reason(completed: subprocess.CompletedProcess[str]) -> str:
    """Pull the human-readable reason out of a failed MinerU run.

    MinerU reports task failures as a JSON blob on one line, which is far too
    noisy to hand a user verbatim - its `error` field is the part that says
    what actually went wrong. Falls back to the last line of output when the
    blob is not JSON, and to the exit code when there is no output at all.
    """
    output = (completed.stderr or "") + "\n" + (completed.stdout or "")
    for line in reversed(output.strip().splitlines()):
        start = line.find("{")
        if start == -1:
            continue
        try:
            payload = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, str) and error.strip():
            return error.strip()

    lines = output.strip().splitlines()
    return lines[-1].strip() if lines else f"exit code {completed.returncode}"


def convert_with_mineru(
    path: Path, source_format: SourceFormat, *, device_mode: str, backend: str,
    model_source: str,
) -> str:
    """Convert a binary document to Markdown with MinerU.

    Settings come from `boepie.config` (the `mineru` section) and are passed
    the way MinerU itself expects them: the device and model source as
    environment variables, the backend as a flag.
    """
    if not mineru_available():
        raise IntakeError(_MINERU_MISSING.format(format=source_format.upper()))

    with tempfile.TemporaryDirectory(prefix="boepie-mineru-") as tmp_name:
        output_dir = Path(tmp_name)
        environment = _mineru_environment(
            device_mode=device_mode, model_source=model_source
        )
        try:
            completed = subprocess.run(
                ["mineru", "-p", str(path), "-o", str(output_dir), "-b", backend],
                capture_output=True, text=True, env=environment, check=False,
            )
        except OSError as error:
            raise IntakeError(f"could not run mineru: {error}") from error

        if completed.returncode != 0:
            raise IntakeError(
                f"mineru failed to convert '{path.name}': "
                f"{_mineru_failure_reason(completed)}"
            )

        # MinerU nests its output under a per-document directory whose exact
        # depth varies by version and backend, so the markdown is located by
        # search rather than by a hardcoded path.
        markdown_files = sorted(output_dir.rglob("*.md"))
        if not markdown_files:
            raise IntakeError(
                f"mineru produced no markdown for '{path.name}'. "
                f"The file may be empty, encrypted, or an unsupported variant."
            )
        largest = max(markdown_files, key=lambda candidate: candidate.stat().st_size)
        return largest.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CHROME_TAGS = ("script", "style", "nav", "header", "footer", "aside", "noscript")


def convert_html(html: str) -> str:
    """Generic HTML to Markdown for an arbitrary web page.

    Not LaTeXML-aware - `boepie.literature.fetch.convert_arxiv_html` handles
    that structured case, where a known container can be selected. An
    arbitrary page has no such container, so this strips chrome instead.
    """
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(html, "html.parser")
    for tag_name in _CHROME_TAGS:
        for node in soup.find_all(tag_name):
            node.decompose()

    body = soup.body or soup
    markdown = markdownify(
        str(body), heading_style="ATX",
        escape_underscores=False, escape_asterisks=False, escape_misc=False,
    )
    return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"


def html_title(html: str) -> str | None:
    from bs4 import BeautifulSoup

    title_tag = BeautifulSoup(html, "html.parser").find("title")
    if title_tag is None or not title_tag.text.strip():
        return None
    return " ".join(title_tag.text.split())


# ---------------------------------------------------------------------------
# The shared core
# ---------------------------------------------------------------------------


def convert_local_file(
    path: Path, *, keep_original: bool, mineru_device_mode: str,
    mineru_backend: str, mineru_model_source: str,
) -> Converted:
    """Convert one local file of any supported format into Markdown."""
    source_format = detect_format(path)
    raw = path.read_bytes()
    checksum = sha256_of(raw)

    if source_format in ("pdf", "docx", "pptx", "xlsx"):
        markdown = convert_with_mineru(
            path, source_format, device_mode=mineru_device_mode,
            backend=mineru_backend, model_source=mineru_model_source,
        )
        via: ConversionVia = "mineru"
    elif source_format == "html":
        markdown = convert_html(read_text_file(path))
        via = "html"
    elif source_format == "code":
        language = _FENCE_LANGUAGES.get(path.suffix.lower(), "")
        body = read_text_file(path)
        # Fenced so the chunker treats it as one block rather than reflowing
        # it as prose, and so a reader can tell code from commentary.
        markdown = f"# {path.name}\n\n```{language}\n{body.rstrip()}\n```\n"
        via = "verbatim"
    else:
        markdown = read_text_file(path)
        via = "verbatim"

    return Converted(
        markdown=markdown,
        via=via,
        format=source_format,
        origin=str(path),
        sha256=checksum,
        original_bytes=raw if keep_original else None,
        original_name=path.name if keep_original else None,
        suggested_title=title_from_markdown(markdown, path.stem),
    )


def convert_url(url: str, *, keep_original: bool) -> Converted:
    """Fetch one web page and convert it to Markdown."""
    try:
        with httpx.Client(
            headers={"User-Agent": _USER_AGENT}, follow_redirects=True
        ) as client:
            response = client.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise IntakeError(f"could not fetch '{url}': {error}") from error

    html = response.text
    markdown = convert_html(html)
    fallback = html_title(html) or urlparse(url).netloc or url
    return Converted(
        markdown=markdown,
        via="html",
        format="html",
        origin=url,
        sha256=sha256_of(response.content),
        original_bytes=response.content if keep_original else None,
        original_name="original.html" if keep_original else None,
        suggested_title=title_from_markdown(markdown, fallback),
    )
