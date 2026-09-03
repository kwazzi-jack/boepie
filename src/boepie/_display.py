# boepie/_display.py
"""Terminal presentation for the CLI: one console, one theme, one place that
knows how boepie's output is coloured.

Everything here adds only colour, weight and style. The wording, the
indentation and the order of the lines belong to the callers and - for search
hits and document spans - to ``boepie.tools._retrieval``, whose renderings the
MCP server emits verbatim. That sharing is why nothing here writes markup into
those payloads: the server's output has to stay plain text, so the CLI styles
the finished string from the outside with a ``RegexHighlighter``. A hit printed
by ``boepie search`` and one returned by ``search_literature`` are the same
bytes, one of them merely wearing ANSI.

Colour disappears on its own when stdout is not a terminal (rich checks
``isatty``), so piping or redirecting any command gives the plain text back
with no flag to remember.

Messages are built as ``rich.text.Text`` rather than markup strings, so an
interpolated title, snippet or config value containing a literal ``[`` is data
rather than an unknown style tag rich would silently swallow.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import click
from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.text import Text
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
#
# Only the eight standard ANSI colours are named, never 256-colour or hex
# values: they resolve against whatever palette the user's terminal is themed
# with, so the output stays legible on a light background as well as a dark
# one.

THEME = Theme(
    {
        # Severities. A short lead word ("Indexed", "Warning:") carries the
        # bold variant; the `.line` variants dress a whole sentence, where
        # bold would shout - see `_line`.
        "success": "bold green",
        "success.line": "green",
        "warning": "bold yellow",
        "warning.line": "yellow",
        "failure": "bold red",
        "failure.line": "red",
        "muted": "dim",
        "muted.line": "dim",
        "heading": "bold",
        "heading.line": "bold",
        # Fragments picked out of ordinary message text.
        "boepie.command": "bold cyan",
        "boepie.option": "cyan",
        "boepie.quoted": "bold",
        "boepie.path": "cyan",
        "boepie.key": "dim",
        "boepie.value": "magenta",
        # Ranked search hits (output family F2).
        "boepie.count": "bold",
        "boepie.query": "yellow",
        "boepie.collection": "bold cyan",
        "boepie.note": "yellow",
        "boepie.rank": "bold cyan",
        "boepie.title": "bold",
        "boepie.section": "italic",
        "boepie.label": "dim",
        "boepie.identifier": "bold magenta",
        "boepie.score_label": "dim",
        "boepie.score": "dim cyan",
        "boepie.chars": "dim",
        # Markdown bodies, shared by hit snippets and read spans.
        "boepie.md_heading": "bold yellow",
        "boepie.md_fence": "dim",
        "boepie.md_code": "cyan",
        "boepie.md_strong": "bold",
        "boepie.md_emphasis": "italic",
        "boepie.md_link": "cyan",
        "boepie.md_url": "underline cyan",
        "boepie.citation": "magenta",
        # `config show`'s TOML.
        "boepie.toml_comment": "dim",
        "boepie.toml_section": "bold cyan",
        "boepie.toml_key": "green",
        "boepie.toml_string": "yellow",
        "boepie.toml_number": "cyan",
        "boepie.toml_bool": "magenta",
    }
)

# highlight=False: rich's default ReprHighlighter guesses at numbers, paths and
# repr syntax in any string printed, which is unpredictable over payloads that
# are already structured. Every renderer below states its own highlighter.
console = Console(theme=THEME, highlight=False)

# Errors follow click's convention of going to stderr, so a piped command's
# stdout stays clean.
error_console = Console(theme=THEME, highlight=False, stderr=True)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
#
# A RegexHighlighter styles each named group as `<base_style><group name>`, and
# applies its patterns in order with later spans drawn over earlier ones. Every
# list below is therefore ordered broad-to-specific: a hit's whole title line
# is styled first, then its section and score fragments are restyled on top.

# Boepie's own top-level subcommands, spelled out so prose that merely mentions
# the program ("boepie is running on built-in defaults") is not mistaken for a
# suggested invocation.
_SUBCOMMANDS = "config|context|corpus|hint|index|read|search|serve|sync"

# A suggested invocation, running to the end of the line or to the quote or
# bracket that closes the aside it sits in. The final character class keeps a
# trailing space out of the styled span.
_COMMAND = rf"(?P<command>\bboepie\s+(?:{_SUBCOMMANDS})\b(?:[^\n'\"`()]*[^\s\n'\"`()])?)"
_OPTION = r"(?P<option>(?<![\w-])--[a-z][\w-]*)"
_QUOTED = r"(?P<quoted>'[^'\n]*'|\"[^\"\n]*\"|`[^`\n]*`)"
# The lookbehind stops a relative path's tail ("literature/bm25") being styled
# from its slash onwards; only a genuine path root starts a match.
_PATH = r"(?P<path>(?<!\w)(?:~|\.{1,2})?/[\w.@+-]+(?:/[\w.@+-]+)*/?)"
# The bracketed alternative keeps a list value (`available=[a, b, c]`) whole;
# without it only the opening item would be styled and the line would read as
# though the highlighting had broken off mid-value.
_KEY_VALUE = r"(?P<key>\b[a-z][\w.]*=)(?P<value>\[[^\]\n]*\]|[^\s,)\]]+)"

_MESSAGE_PATTERNS = [_PATH, _KEY_VALUE, _QUOTED, _COMMAND, _OPTION]

# Markdown structure in a document body. Deliberately lightweight: the markers
# are styled where they stand, never consumed, because the CLI shows the same
# characters the corpus holds and an agent would read.
_MD_HEADING = r"(?m)^(?P<md_heading>[ \t]*#{1,6}[ \t][^\n]*)$"
_MD_FENCE = r"(?m)^(?P<md_fence>[ \t]*(?:```|~~~)[^\n]*)$"
_MD_CODE = r"(?P<md_code>`[^`\n]+`)"
_MD_STRONG = r"(?P<md_strong>\*\*[^*\n]+\*\*)"
# The word-boundary guards keep `snake_case` and 3*4 out of the emphasis span.
_MD_EMPHASIS = r"(?P<md_emphasis>(?<![\w*_])[*_][^\s*_][^*_\n]*[*_](?![\w*_]))"
_MD_LINK = r"(?P<md_link>\[[^\]\n]*\])(?P<md_url>\([^)\n]*\))"
_URL = r"(?P<md_url>\bhttps?://[^\s)\]>,]+)"

# Citations, the one piece of domain syntax worth picking out of paper prose.
# The bracket form's lookbehind requires a preceding character on the line, so
# a hit's `[1]` rank marker - always at column zero - is never mistaken for a
# numbered reference.
_CITATION_BRACKET = r"(?<=[^\n])(?P<citation>\[(?:\d+(?:\s*[,-]\s*\d+)*|@[\w:./-]+)\])"
_CITATION_ARXIV = (
    r"(?P<citation>\b(?:arXiv:)?\d{4}\.\d{4,5}(?:v\d+)?\b"
    r"|\b(?:arXiv:)?[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?\b)"
)
_CITATION_DOI = r"(?P<citation>\b(?:doi:)?10\.\d{4,9}/[^\s,)\]]+)"
# (Smith 2020), (Smith & Jones 2020), (Smith et al. 2020; Jones 2021).
_CITATION_AUTHOR_YEAR = (
    r"(?P<citation>\(\s*[A-Z][^()\n]{0,80}?\b\d{4}[a-z]?"
    r"(?:\s*;\s*[^()\n]{0,80}?\b\d{4}[a-z]?)*\s*\))"
)
# Smith et al. (2020), Smith & Jones (2020).
_CITATION_NARRATIVE = (
    r"(?P<citation>\b[A-Z][\w'-]+(?:\s+(?:et\s+al\.|&\s+[A-Z][\w'-]+))?\s+\(\d{4}[a-z]?\))"
)

# Block-level markdown is only meaningful where the CLI prints a document's
# real line structure, which is spans but not hits: `format_hits`'s default
# `short` snippet collapses a chunk onto one line, so a chunk that merely
# begins with `##` or a fence would have its entire snippet styled as a
# heading or a code block.
_BLOCK_PATTERNS = [_MD_FENCE, _MD_HEADING]

# Inline markdown and citations survive that collapsing, so they apply to
# snippets and spans alike.
_INLINE_PATTERNS = [
    _MD_EMPHASIS,
    _MD_STRONG,
    _MD_CODE,
    _URL,
    _MD_LINK,
    _CITATION_AUTHOR_YEAR,
    _CITATION_NARRATIVE,
    _CITATION_ARXIV,
    _CITATION_DOI,
    _CITATION_BRACKET,
]

# `format_hits`: a header, then per hit a `[rank] title #section  scores` line
# followed by indented `read:`/`source:` handles and an optional snippet.
_HITS_HEADER = r'(?m)^(?P<count>\d+) hits for (?P<query>".*") in (?P<collection>\S+)$'
_HITS_NOTE = r"(?m)^(?P<note>Note: [^\n]*)$"
_HIT_LINE = (
    r"(?m)^(?P<rank>\[\d+\])\s(?P<title>[^\n]*?)(?=\s#\S|\s\s(?:bm25|cos|rrf)=|$)"
)
_HIT_SECTION = (
    r"(?m)^\[\d+\]\s[^\n]*?(?P<section>\s#\S[^\n]*?)(?=\s\s(?:bm25|cos|rrf)=|$)"
)
_HIT_SCORE = r"(?P<score_label>\b(?:bm25|cos|rrf)=)(?P<score>[\d.]+)"
_HIT_LABEL = r"(?m)^\s+(?P<label>read|source):"
_HIT_HANDLE = r"(?P<key>\b(?:document_id|chunk_index)=)(?P<identifier>\S+)"
_HIT_SOURCE = r"(?m)^\s+source:\s(?P<path>[^\n]+?)(?=\s\(chars\s|$)"
_HIT_CHARS = r"(?P<chars>\(chars \d+-\d+\))"

# `format_span`: one `document_id=... chunks=... chars=...` line, `source:`,
# an optional `sections:`, then the document text.
_SPAN_HANDLE = r"(?P<key>\b(?:document_id|chunks|chars)=)(?P<identifier>\S+)"
_SPAN_LABEL = r"(?m)^(?P<label>source|sections):"
_SPAN_SOURCE = r"(?m)^source:\s(?P<path>[^\n]+)$"
_SPAN_SECTIONS = r"(?m)^sections:\s(?P<section>[^\n]+)$"

# `config show`: TOML, which rich has no highlighter for.
_TOML_COMMENT = r"(?m)(?P<toml_comment>(?:^|\s\s)#[^\n]*)$"
_TOML_SECTION = r"(?m)^(?P<toml_section>\[[\w.-]+\])$"
_TOML_KEY = r"(?m)^(?P<toml_key>[\w.-]+)(?=\s=\s)"
_TOML_STRING = r"(?P<toml_string>\"(?:[^\"\\\n]|\\.)*\")"
_TOML_NUMBER = r"(?<== )(?P<toml_number>-?\d+(?:\.\d+)?)\b"
_TOML_BOOL = r"(?<== )(?P<toml_bool>true|false)\b"


class MessageHighlighter(RegexHighlighter):
    """Picks the actionable parts out of an ordinary CLI line.

    Runs over every message the CLI prints, so a suggested command, a path or
    a `key=value` is findable at a glance without each call site marking it up
    by hand.
    """

    base_style = "boepie."
    highlights = _MESSAGE_PATTERNS


class HitHighlighter(RegexHighlighter):
    """Styles `format_hits` output without touching a character of it."""

    base_style = "boepie."
    highlights = [
        *_INLINE_PATTERNS,
        _HITS_HEADER,
        _HITS_NOTE,
        _HIT_LINE,
        _HIT_SECTION,
        _HIT_SCORE,
        _HIT_LABEL,
        _HIT_HANDLE,
        _HIT_SOURCE,
        _HIT_CHARS,
    ]


class SpanHighlighter(RegexHighlighter):
    """Styles `format_span` output without touching a character of it."""

    base_style = "boepie."
    highlights = [
        *_BLOCK_PATTERNS,
        *_INLINE_PATTERNS,
        _SPAN_HANDLE,
        _SPAN_LABEL,
        _SPAN_SOURCE,
        _SPAN_SECTIONS,
    ]


class TomlHighlighter(RegexHighlighter):
    """Styles the TOML `config show` prints."""

    base_style = "boepie."
    highlights = [
        _TOML_SECTION,
        _TOML_KEY,
        _TOML_STRING,
        _TOML_NUMBER,
        _TOML_BOOL,
        _TOML_COMMENT,
    ]


_MESSAGE_HIGHLIGHTER = MessageHighlighter()
_HIT_HIGHLIGHTER = HitHighlighter()
_SPAN_HIGHLIGHTER = SpanHighlighter()
_TOML_HIGHLIGHTER = TomlHighlighter()


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _line(style: str | None, text: str, lead: str | None, indent: str) -> Text:
    """One message line, styled by the `lead`-carries-the-colour convention.

    With a `lead` ("Indexed", "Warning:") only that word takes `style` and the
    rest of the sentence stays plain; without one the whole line takes the
    quieter `.line` variant, since a full sentence in bold colour shouts.
    Either way the text is highlighted, never parsed as markup.
    """
    prefix = indent if lead is None else f"{indent}{lead}{' ' if text else ''}"
    line = Text(f"{prefix}{text}")
    _MESSAGE_HIGHLIGHTER.highlight(line)
    if lead is None:
        if style is not None:
            line.style = f"{style}.line"
    elif style is not None:
        line.stylize(style, len(indent), len(indent) + len(lead))
    return line


def info(text: str = "", *, lead: str | None = None, indent: str = "") -> None:
    """A neutral line: no severity, only the fragments the highlighter finds."""
    console.print(_line(None, text, lead, indent))


def success(text: str = "", *, lead: str | None = None, indent: str = "") -> None:
    """Something happened. `lead` is the past-tense verb that says what."""
    console.print(_line("success", text, lead, indent))


def warning(text: str = "", *, lead: str | None = None, indent: str = "") -> None:
    """Something is off but the command carries on."""
    console.print(_line("warning", text, lead, indent))


def error(text: str = "", *, lead: str | None = None, indent: str = "") -> None:
    """Something failed. Printed to stdout like the rest of a command's
    report; a failure that aborts the command raises `CliError` instead."""
    console.print(_line("failure", text, lead, indent))


def heading(text: str = "", *, lead: str | None = None, indent: str = "") -> None:
    """A line that names the thing the lines under it are about."""
    console.print(_line("heading", text, lead, indent))


def muted(text: str = "", *, lead: str | None = None, indent: str = "") -> None:
    """An aside worth printing but not worth reading first."""
    console.print(_line("muted", text, lead, indent))


# Set only by `following_steps(False)`. A composite command runs the steps
# its parts would otherwise be advising, and a sub-command has no way to
# know it is not the outermost caller.
_next_steps_wanted = True


@contextlib.contextmanager
def following_steps(wanted: bool) -> Iterator[None]:
    """Suppress the `Next:` advice inside the block when `wanted` is False.

    `corpus fetch` closes by telling you to run `index build`. Inside
    `boepie setup` that is advice to do what the next phase does anyway, and
    twice over for two collections. Suppressing it here keeps the decision
    with the command that knows it is a composite, rather than teaching four
    sub-commands to ask whether anyone is above them.
    """
    global _next_steps_wanted
    previous = _next_steps_wanted
    _next_steps_wanted = wanted
    try:
        yield
    finally:
        _next_steps_wanted = previous


def next_step(
    command: str, *, before: str = "", note: str = "", indent: str = ""
) -> None:
    """The "what to run now" line a command closes with.

    One helper so the phrasing and the command's styling stay identical
    everywhere; six call sites used to spell it out by hand.
    """
    if not _next_steps_wanted:
        return
    preamble = f"{before} " if before else ""
    trailer = f" {note}" if note else ""
    info(f"{preamble}Next: {command}{trailer}", indent=indent)


def path(location: object) -> None:
    """A bare filesystem path on its own line.

    soft_wrap because a path is one token: rich's word wrap would otherwise
    break a long one mid-token into something that cannot be copied out.
    """
    console.print(Text(str(location), style="boepie.path"), soft_wrap=True)


def plain(text: str) -> None:
    """Text that must reach stdout exactly as given, with no styling at all -
    a config value being read by a script, for instance."""
    console.print(Text(text), soft_wrap=True)


# ---------------------------------------------------------------------------
# Payload renderers
# ---------------------------------------------------------------------------


def hits(payload: str) -> None:
    """Ranked search results, exactly as the MCP tools render them."""
    console.print(_HIT_HIGHLIGHTER(payload))


def span(payload: str) -> None:
    """One document span, exactly as the MCP read_* tools render it."""
    console.print(_SPAN_HIGHLIGHTER(payload))


def toml(payload: str) -> None:
    """`config show`'s resolved settings.

    soft_wrap: the output is valid TOML and stays that way only if rich leaves
    the lines alone - a wrapped comment or path would not survive being piped
    to a file.
    """
    console.print(_TOML_HIGHLIGHTER(payload), soft_wrap=True)


def hint(coordinate: str, snippet: str) -> None:
    """One `path#section: snippet` coordinate line from `boepie hint`."""
    line = Text()
    line.append(coordinate, style="boepie.identifier")
    line.append(": ")
    line.append(snippet, style="muted")
    console.print(line, soft_wrap=True)


def _document_handle(document_id: str, managed_by: str) -> Text:
    """The `(id=..., who)` trailer: what `read_*` and `corpus remove` take,
    and who manages the document."""
    handle = Text("(", style="muted")
    handle.append("id=", style="boepie.key")
    handle.append(document_id, style="boepie.identifier")
    handle.append(", ", style="muted")
    handle.append(*_managed_marker(managed_by))
    handle.append(")", style="muted")
    return handle


def document_entry(title: str, document_id: str, managed_by: str) -> None:
    """Print one `corpus list` entry, keeping its `(id=..., ...)` handle whole.

    Paper titles routinely outrun a terminal, and rich wrapping the composed
    line breaks wherever a space falls - which lands mid-handle and leaves
    `boepie)` stranded on a line of its own. So the title is wrapped alone and
    the handle is placed after it only if it fits, otherwise on the
    continuation line, where it is still visibly part of this entry.
    """
    indent = "  "
    handle = _document_handle(document_id, managed_by)
    width = max(console.width - len(indent), 20)
    wrapped = Text(title or document_id, style="boepie.title").wrap(console, width)

    lines: list[Text] = [Text(line.plain) for line in wrapped] or [Text()]
    for line, source in zip(lines, wrapped):
        line.spans = list(source.spans)
        line.style = source.style
    if len(lines[-1]) + 1 + len(handle) <= width:
        lines[-1].append(" ")
        lines[-1].append_text(handle)
    else:
        lines.append(handle)

    for position, line in enumerate(lines):
        console.print(
            line if position == 0 else Text(indent).append_text(line), soft_wrap=True
        )


def document_leaf(title: str, document_id: str, managed_by: str) -> Text:
    """`corpus tree`'s leaf label. Same facts as `document_entry`, laid out for
    a tree: boepie-managed is the norm here, so only `yours` is called out."""
    line = Text()
    line.append(title, style="boepie.title")
    line.append(f" {document_id}", style="boepie.identifier")
    if managed_by != "boepie":
        line.append(" (yours)", style="warning.line")
    return line


def group_leaf(name: str) -> Text:
    """`corpus tree`'s branch label for a user-created group directory."""
    return Text(f"{name}/", style="heading")


def collection_root(collection: str, location: object) -> Text:
    """A collection and where it lives on disk.

    Shared by `corpus tree`'s root label and `corpus status`'s per-collection
    heading, so the two commands name a collection the same way. Callers
    print it with `soft_wrap`: the path is one token, and rich's word wrap
    would otherwise break a long one mid-token into something uncopyable.
    """
    root = Text()
    root.append(collection, style="boepie.collection")
    root.append(f" {location}", style="boepie.path")
    return root


def _managed_marker(managed_by: str) -> tuple[str, str]:
    """The word and style that say who a document belongs to."""
    if managed_by == "boepie":
        return "boepie", "success.line"
    if managed_by == "user":
        return "yours", "warning.line"
    return managed_by, "muted"


# ---------------------------------------------------------------------------
# Errors that abort the command
# ---------------------------------------------------------------------------


class CliError(click.ClickException):
    """A ClickException that reports through the themed stderr console.

    click's own `show` writes an unstyled `Error: ...`; this keeps that
    wording and stream and only colours it, so scripts parsing stderr see
    what they always did.

    soft_wrap for the same reason: click emits the message as one unwrapped
    line, and these messages routinely name the command that fixes the
    problem - rich's word wrap would split that across a line break and make
    it uncopyable.
    """

    def show(self, file: object = None) -> None:
        # Text(style=...) would make `failure` the base style of everything
        # appended after it, not just the prefix.
        line = Text()
        line.append("Error: ", style="failure")
        message = Text(self.format_message())
        _MESSAGE_HIGHLIGHTER.highlight(message)
        line.append_text(message)
        error_console.print(line, soft_wrap=True)
