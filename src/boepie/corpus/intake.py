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
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
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

# Every suffix boepie has a converter for. This is the accept-list a folder
# walk works from, and it exists because `detect_format` below deliberately
# does not: asked about an unknown suffix it answers "code", which is right
# when a user named one file explicitly and vouched for it, and catastrophic
# when walking a directory. `read_text_file`'s encoding ladder ends in latin-1
# and so never fails, meaning a naive walk would ingest an ELF binary, a .pyc
# or a git pack file as a fenced block of mojibake, silently, and index it.
#
# Derived from the sets above rather than restated, so a format added to one
# of them is walkable without a second edit.
SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    _MARKDOWN_SUFFIXES | _TEXT_SUFFIXES | _HTML_SUFFIXES | _CODE_SUFFIXES
) | frozenset(_MINERU_SUFFIXES)


def is_supported_suffix(path: Path, extra: Sequence[str] = ()) -> bool:
    """Whether a folder walk should pick `path` up.

    `extra` is `corpus.extra_file_types`, added rather than substituted: the
    common need is one more extension (a `.ipynb`, a house format), not a
    replacement for the sixty boepie already knows.
    """
    suffix = path.suffix.lower()
    if not suffix:
        # No extension at all. A `Makefile` is real text, but so is every
        # extensionless binary, and a walk cannot tell them apart by name.
        return False
    return suffix in SUPPORTED_SUFFIXES or suffix in {
        item.lower() if item.startswith(".") else f".{item.lower()}" for item in extra
    }


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

# The formats that have to go through MinerU rather than being read as text.
MINERU_FORMATS: frozenset[SourceFormat] = frozenset(_MINERU_SUFFIXES.values())


def mineru_available() -> bool:
    return shutil.which("mineru") is not None


def require_mineru(paths: Sequence[Path]) -> None:
    """Fail now if MinerU is needed and missing, rather than once per file.

    A folder of fifty PDFs would otherwise produce fifty copies of the same
    install instructions, and would produce them one at a time over the course
    of a run that cannot succeed.
    """
    if not paths or mineru_available():
        return
    formats = sorted({path.suffix.lstrip(".").upper() for path in paths})
    raise IntakeError(_MINERU_MISSING.format(format="/".join(formats)))


def mineru_no_markdown(path: Path, reason: str | None) -> str:
    """Why one document came back from MinerU with nothing.

    `reason` is MinerU's own wording when the process itself failed; without
    one the document simply produced no output, which is what an empty,
    encrypted or unsupported file looks like from here.
    """
    if reason:
        return f"mineru failed to convert '{path.name}': {reason}"
    return (
        f"mineru produced no markdown for '{path.name}'. "
        f"The file may be empty, encrypted, or an unsupported variant."
    )


# Block kinds MinerU classifies as page furniture rather than body text.
# These are exactly what the rendered Markdown throws away, and exactly where
# a paper stamps its own identity: `arXiv:1805.03410v2 [astro-ph.IM]` arrives
# as `page_aside_text`, a journal DOI as a `page_footer`, an ADS bibcode on a
# scanned paper as a `page_header`.
_FURNITURE_TYPES = frozenset({
    "page_header", "page_footer", "page_footnote", "page_aside_text",
    "page_number",
})


def _v2_block_text(block: dict[str, Any]) -> str:
    """The plain text of one v2 content block, whatever kind it is.

    v2 nests a block's pieces under a key named after the block's own type
    (`{"type": "page_header", "content": {"page_header_content": [...]}}`),
    so the key has to be built from the type rather than looked up by name.
    """
    kind = str(block.get("type", ""))
    content = block.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, dict):
        return ""
    pieces = content.get(f"{kind}_content")
    if not isinstance(pieces, list):
        return ""
    return " ".join(
        str(piece.get("content", ""))
        for piece in pieces
        if isinstance(piece, dict)
    )


def _front_page_of_v2(pages: list[Any]) -> str:
    """First-page text from a `_content_list_v2.json`, furniture first."""
    furniture: list[str] = []
    body: list[str] = []
    for block in pages[0] if pages and isinstance(pages[0], list) else []:
        if not isinstance(block, dict):
            continue
        text = _v2_block_text(block)
        if not text:
            continue
        target = furniture if block.get("type") in _FURNITURE_TYPES else body
        target.append(text)
    return "\n".join(furniture + body)


def _front_page_of_v1(blocks: list[Any]) -> str:
    """First-page text from the older flat `_content_list.json`.

    v1 has no notion of furniture - every block is `text` or `image` with a
    `page_idx` - so the ordering v2 allows is not available here. It still
    carries the aside the Markdown drops, which is the part that matters.
    """
    return "\n".join(
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, dict)
        and block.get("page_idx") == 0
        and block.get("text")
    )


def front_page_text(markdown_path: Path) -> str:
    """Everything MinerU read off page one, including what it did not render.

    A paper's own identifier is systematically *absent* from the converted
    Markdown - verified on real conversions, `grep -c arxiv` is 0 - because
    MinerU classifies the `arXiv:...` stamp as page furniture and furniture is
    not body text. It survives in the content list beside the Markdown, which
    is the only reason a local PDF can be identified at all.

    Only page one. A bibliography offers dozens of other people's identifiers
    and every one of them is a wrong answer, so the search is confined to the
    page where a paper states its own.
    """
    for suffix in ("_content_list_v2.json", "_content_list.json"):
        path = markdown_path.with_name(f"{markdown_path.stem}{suffix}")
        if not path.is_file():
            continue
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, list):
            continue
        return (
            _front_page_of_v2(parsed)
            if suffix.endswith("_v2.json")
            else _front_page_of_v1(parsed)
        )
    return ""


@dataclass(frozen=True)
class MineruResult:
    """What one MinerU process produced.

    Reported per document rather than as one pass or fail: MinerU converts
    the documents it can and records the rest in its own task log, so a path
    missing from `markdown` is that document's failure alone and must not
    cost the others. `failure_reason` is MinerU's wording when the process
    exited non-zero, which is the only context a bare "produced nothing"
    would otherwise lack.
    """

    markdown: dict[Path, str]
    # Page-one text per document, furniture included. Carried out of the
    # temporary directory because that is the only place it exists: the
    # content list is discarded with the run, and the Markdown never had it.
    front_page: dict[Path, str] = field(default_factory=dict)
    failure_reason: str | None = None


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


# How much of the original filename a staged copy keeps. The index prefix is
# what makes the name unique; the rest is there only so MinerU's own error
# output names something the user recognises, and truncating is easier than
# discovering each filesystem's own limit.
_STAGED_STEM_LIMIT = 100


def _stage_for_mineru(paths: Sequence[Path], staged_dir: Path) -> dict[str, Path]:
    """Present a batch to MinerU as one directory, and map its output back.

    `-p` takes a single path, so several documents can only be handed over as
    a directory. They are renamed on the way in because MinerU names its
    output directory after the input's stem: two `README.pdf` from different
    folders would otherwise write to the same place and one would silently
    win. Hard-linked where the filesystem allows it, since the alternative is
    copying every source into the temporary directory.
    """
    staged: dict[str, Path] = {}
    for index, path in enumerate(paths):
        target = staged_dir / (
            f"{index:04d}-{path.stem[:_STAGED_STEM_LIMIT]}{path.suffix.lower()}"
        )
        try:
            os.link(path, target)
        except OSError:
            # A different filesystem, or one without hard links.
            shutil.copy2(path, target)
        staged[target.stem] = path
    return staged


def convert_with_mineru(
    paths: Sequence[Path], *, device_mode: str, backend: str, model_source: str,
    page_limit: int | None = None,
) -> MineruResult:
    """Convert binary documents to Markdown with MinerU, in one process.

    Every path shares one invocation, and that is the whole point of taking a
    sequence: MinerU spends around twenty seconds loading its model stack
    before it converts anything, so a process per document pays that toll
    every time. Measured on a 19-page paper: 41s for one document alone
    against 88s for four in one call.

    The caller decides how large a run is - see `boepie.corpus.add` - because
    MinerU writes nothing until the whole run finishes, so a run is also the
    unit of progress and the unit lost to an interruption.

    `page_limit` stops MinerU after that many pages. Used to survey a batch
    before committing to converting it: a paper states its own identity on
    page one, so two pages are enough to learn what every document in a folder
    *is*, at about 1.2 seconds each against 16 for a full conversion. The
    markdown that comes back is a fragment and is meant to be discarded; the
    front page is the point.

    Settings come from `boepie.config` (the `mineru` section) and are passed
    the way MinerU itself expects them: the device and model source as
    environment variables, the backend as a flag.
    """
    if not paths:
        return MineruResult(markdown={})
    require_mineru(paths)

    with tempfile.TemporaryDirectory(prefix="boepie-mineru-") as tmp_name:
        staged_dir = Path(tmp_name) / "input"
        output_dir = Path(tmp_name) / "output"
        staged_dir.mkdir()
        output_dir.mkdir()
        staged = _stage_for_mineru(paths, staged_dir)

        environment = _mineru_environment(
            device_mode=device_mode, model_source=model_source
        )
        command = ["mineru", "-p", str(staged_dir), "-o", str(output_dir), "-b", backend]
        if page_limit is not None:
            command += ["-s", "0", "-e", str(page_limit - 1)]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, env=environment, check=False,
            )
        except OSError as error:
            raise IntakeError(f"could not run mineru: {error}") from error

        markdown: dict[Path, str] = {}
        front_page: dict[Path, str] = {}
        for stem, original in staged.items():
            # MinerU nests each document's output under a directory named
            # after its input stem, at a depth that varies by version and
            # backend, so the markdown is found by searching that directory
            # rather than by a hardcoded path.
            produced = sorted((output_dir / stem).rglob("*.md"))
            if not produced:
                continue
            largest = max(produced, key=lambda candidate: candidate.stat().st_size)
            markdown[original] = largest.read_text(encoding="utf-8", errors="replace")
            front_page[original] = front_page_text(largest)

        reason = (
            _mineru_failure_reason(completed) if completed.returncode != 0 else None
        )
        if not markdown and reason is not None:
            raise IntakeError(f"mineru failed: {reason}")
        return MineruResult(
            markdown=markdown, front_page=front_page, failure_reason=reason
        )


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
    prepared_markdown: str | None = None,
) -> Converted:
    """Convert one local file of any supported format into Markdown.

    `prepared_markdown` is MinerU output an earlier batch already produced for
    this path (see `boepie.corpus.add._plan_binaries`). Supplying it is what
    keeps a batched conversion from being repeated a document at a time; it is
    ignored for the formats MinerU never sees.
    """
    source_format = detect_format(path)
    raw = path.read_bytes()
    checksum = sha256_of(raw)

    if source_format in MINERU_FORMATS:
        markdown = prepared_markdown
        if markdown is None:
            result = convert_with_mineru(
                [path], device_mode=mineru_device_mode,
                backend=mineru_backend, model_source=mineru_model_source,
            )
            markdown = result.markdown.get(path)
            if markdown is None:
                raise IntakeError(mineru_no_markdown(path, result.failure_reason))
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
