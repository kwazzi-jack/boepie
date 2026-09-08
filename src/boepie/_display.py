# boepie/_display.py
"""Terminal presentation for the CLI: one console, one theme, one place that
knows how boepie's output is shaped.

**The report grammar.** Every line a command prints about work it did is an
*operation*: a past-tense verb at the margin, then the thing it acted on,
then how long it took. What belongs to an operation - the documents it
touched, the command to run next - is indented one step under it
(``_CONTENT_INDENT``). Read down the left edge and you have the order things
happened in::

    Initialized bundle at .boepie in 7ms
    Created 3 corpus directories in 0ms
      + /home/brian/.local/share/boepie/literature
    Rebuilt literature index - 2 of 19 documents added in 1m11s
    Registered boepie with claude, copilot
      hint: run `boepie register --force`

**Indentation, not alignment.** boepie used to right-align its verbs into an
11-character column, which is uv's and cargo's convention and which its
longest verbs happened to fit exactly. That gave the payload one column but
cost two left edges: a detail marker cannot join the verb column without
being inset ten characters, so the eye had to track both column 3 and column
12. With everything at the margin there is one edge for operations and one
for their contents. The payload no longer starts in a fixed column, and the
in-flight/finished pair (``Indexing`` -> ``Indexed``) now shares a *start*
rather than an end - which still replaces cleanly, since the progress block
is erased whole.

Diagnostics are the exception. ``warning:`` and ``error:`` stay at column
zero: a problem has to break the left edge to be seen, and must not scan as
one more step that went fine. ``hint:`` is indented with the other content,
because advice is subordinate to the line it follows.

**Two shapes, and only two.** A command either reports a *sequence of
operations* or describes *state*, and the two want different layouts:

- ``setup``, ``sync``, ``corpus sync``/``add``/``remove``/``move`` and
  ``corpus index`` do things in order, so they use the operation lines above.
- ``corpus status``, ``index status``, ``corpus list`` and ``corpus tree``
  answer "what is here", so they keep an aligned ``label:`` column
  (``_STATUS_LABEL_WIDTH`` in ``cli``): several facts about several things,
  where the label carries the severity and one column can be scanned for
  anything wrong.

Forcing the second group into operation lines would mean writing ``Checked``
against every row of a status listing, claiming an action where there was
only a look. The distinction is the design, not an unconverted remnant - what
the two share is diction, not layout, and both say what is true in the same
plain words.

**This module owns layout, not wording.** It decides where a verb sits and
what colour it wears; which verb, and the sentence after it, belong to the
caller. The one exception is by design: search hits and document spans come
from ``boepie.tools._retrieval``, whose renderings the MCP server emits
verbatim. The server's output has to stay plain text, so nothing writes markup
into those payloads - the CLI styles the finished string from the outside with
a ``RegexHighlighter``. A hit printed by ``boepie search`` and one returned by
``search_literature`` are the same bytes, one of them merely wearing ANSI.

Colour disappears on its own when stdout is not a terminal (rich checks
``isatty``), so piping or redirecting any command gives the plain text back
with no flag to remember.

Messages are built as ``rich.text.Text`` rather than markup strings, so an
interpolated title, snippet or config value containing a literal ``[`` is data
rather than an unknown style tag rich would silently swallow.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Sequence

import click
from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.padding import Padding
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
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
        # One style per marker, as uv colours them: what happened is in the
        # glyph, so the name beside it can stay quiet. `=` is unchanged and
        # gets no colour at all - it is the absence of news.
        # Not `muted`: a hint is the one line a reader is meant to act on,
        # and dim made it recede into the report it follows.
        "boepie.hint": "cyan",
        "boepie.spinner": "cyan",
        # rich's own progress styles, restated in the eight standard ANSI
        # colours. Its defaults are a truecolor magenta gradient and a green
        # count, which is the one place boepie's output stopped resolving
        # against the user's palette - unreadable on some light themes and
        # nothing like the rest of the report.
        "bar.back": "bright_black",
        "bar.complete": "cyan",
        "bar.finished": "green",
        "bar.pulse": "cyan",
        "progress.download": "dim",
        "progress.elapsed": "dim",
        "progress.remaining": "dim",
        "boepie.added": "green",
        "boepie.removed": "red",
        "boepie.changed": "yellow",
        "boepie.marker": "bold",
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

# Progress bars go to stderr so a redirected stdout keeps only the report.
_progress_console = Console(theme=THEME, highlight=False, stderr=True)

# Advances a bar: no arguments steps it by one, `(done, total)` sets both.
type ProgressUpdate = Callable[..., None]


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
#
# A RegexHighlighter styles each named group as `<base_style><group name>`, and
# applies its patterns in order with later spans drawn over earlier ones. Every
# list below is therefore ordered broad-to-specific: a hit's whole title line
# is styled first, then its section and score fragments are restyled on top.

# A command to run. The backtick is the marker, and `command()` below is the
# only thing that writes one, so the highlighter never has to work out which
# words in a sentence are an invocation.
#
# It used to work that out from a hardcoded list of boepie's own subcommands,
# and the list silently fell behind the CLI: `setup` was added as a command
# and not to the list, so `boepie setup` rendered in the plain quoted style
# while `boepie context init` beside it rendered cyan - two suggestions in
# two colours for no reason a reader could see. A marker cannot drift that
# way, and it also covers the commands that are not boepie's own
# (`uv run scripts/...`), which the list could never have matched.
_COMMAND = r"(?P<command>`[^`\n]+`)"
_OPTION = r"(?P<option>(?<![\w-])--[a-z][\w-]*)"
# A value rather than a command: a config key, an id, a citekey, something
# the user typed or boepie read back. Quotes only - backticks now mean
# something else, and the two shared one style while they shared one pattern.
_QUOTED = r"(?P<quoted>'[^'\n]*'|\"[^\"\n]*\")"
# The lookbehind stops a relative path's tail ("literature/bm25") being styled
# from its slash onwards; only a genuine path root starts a match.
_PATH = r"(?P<path>(?<!\w)(?:~|\.{1,2})?/[\w.@+-]+(?:/[\w.@+-]+)*/?)"
# The bracketed alternative keeps a list value (`available=[a, b, c]`) whole;
# without it only the opening item would be styled and the line would read as
# though the highlighting had broken off mid-value.
_KEY_VALUE = r"(?P<key>\b[a-z][\w.]*=)(?P<value>\[[^\]\n]*\]|[^\s,)\]]+)"

# _COMMAND last: a command is one thing and must render as one span, so it
# draws over the option and path fragments inside it. `--collection` styled
# on top of the invocation it belongs to lost the command's bold and made a
# single suggestion look like two.
_MESSAGE_PATTERNS = [_PATH, _KEY_VALUE, _QUOTED, _OPTION, _COMMAND]

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


# ---------------------------------------------------------------------------
# The operation column
# ---------------------------------------------------------------------------

# What an operation's contents are indented by: its detail lines, and the
# hint that follows it. Two levels and no alignment - an operation at the
# margin, everything belonging to it one step in.
#
# boepie used to right-align its verbs into an 11-character column, uv's own
# value. That gave the payload one column but cost two left edges, since a
# detail marker cannot join the verb column without being inset ten
# characters; the indentation carries the structure now, and nothing is
# aligned to a width that a longer verb could overflow.
_CONTENT_INDENT = 2

# Detail lines: one marker character, then the item. `uv` prints ` + rich==15.0.0`.
_DETAIL_INDENT = " " * _CONTENT_INDENT

# How many details an operation prints before summarising the rest. A first
# docs fetch touches 98 pages, and 98 lines of page titles is a wall nobody
# reads; the count above it already said how many there were. `--verbose`
# lifts this.
DETAIL_LIMIT = 10


def _format_elapsed(seconds: float) -> str:
    """`31ms`, `4.7s`, `2m25s` - the precision a reader can act on.

    Milliseconds below a second, as uv prints them: `in 0.0s` on work that
    took 40ms reads as a rounding artefact and tells the reader nothing,
    where `40ms` says plainly that the step is free. Past a minute the
    seconds still matter (a 2m25s fetch and a 2m55s one feel different) but
    tenths do not.
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m{remainder:02d}s"


def using(text: str) -> None:
    """The environment banner a command opens with.

    It names the ground the run stands on rather than something boepie did,
    and now that operations start at the margin too, that is carried by the
    dim `Using` lead alone rather than by position - every other line's first
    word is a past-tense verb in a severity colour.
    """
    if _quiet:
        return
    line = Text(f"Using {text}")
    _MESSAGE_HIGHLIGHTER.highlight(line)
    line.stylize("muted", 0, 5)
    console.print(line, soft_wrap=True)


def operation(
    verb: str, text: str = "", *, elapsed: float | None = None, style: str = "success"
) -> None:
    """One line of the report: what boepie did, to what, and how long it took.

    The verb starts at the margin and carries the colour; the rest stays
    plain, so a scan down the left edge reads as a list of operations in the
    order they happened. `elapsed`, when given, is appended as `in 1.4s` -
    the answer to "has this stopped, or is it just slow", which a reader can
    only learn by being told what a step normally costs.

    **Left-aligned, not right-aligned into a column.** uv and cargo pad their
    verbs to a fixed width so the payload lands in one column, and boepie
    copied that; the cost is two left edges, because a detail line's marker
    cannot sit in the verb's column without being inset ten characters. With
    everything at the margin there is one edge for operations and one for
    their contents, which is what the indentation now carries. The payload no
    longer starts in a fixed column - `Rebuilt literature...` and `Registered
    boepie...` begin three characters apart - and that was the trade made
    knowingly.
    """
    if _quiet:
        return
    trailer = f" in {_format_elapsed(elapsed)}" if elapsed is not None else ""
    line = Text(f"{verb} {text}{trailer}".rstrip())
    _MESSAGE_HIGHLIGHTER.highlight(line)
    line.stylize(style, 0, len(verb))
    console.print(line, soft_wrap=True)


# What each detail marker means, and so how it is coloured.
_MARKER_STYLES = {
    "+": "boepie.added",
    "-": "boepie.removed",
    "~": "boepie.changed",
}


def detail(marker: str, text: str) -> None:
    """One item an operation touched: `+` added, `-` removed, `~` changed.

    **The marker carries the colour and the name stays dim.** A detail line
    is a list of things, and what a reader scans for is which kind each one
    is - that is one character, in the same column every time. Printing the
    names at full weight made a first docs fetch's ten lines shout as loudly
    as the operation they belong to.

    soft_wrap because these are overwhelmingly paths and identifiers - one
    token each, which rich's word wrap would break in the middle.
    """
    if _quiet:
        return
    # Assembled rather than styled over a base: a `muted` base is `dim`, and
    # rich composes styles, so a green marker on top of it comes out dim
    # green - which is precisely the colour that was meant to stand out.
    line = Text.assemble(
        _DETAIL_INDENT,
        (marker, _MARKER_STYLES.get(marker, "boepie.marker")),
        (f" {text}", "muted"),
    )
    console.print(line, soft_wrap=True)


def details(marker: str, items: Sequence[str], *, limit: int | None = DETAIL_LIMIT) -> None:
    """Every item an operation touched, capped so a large batch stays readable.

    `limit=None` prints all of them, which is what `--verbose` passes. The
    count is already on the operation line above, so the elision loses nothing
    but names.
    """
    shown = list(items) if limit is None else list(items)[:limit]
    for item in shown:
        detail(marker, item)
    remaining = len(items) - len(shown)
    if remaining:
        muted(f"{_DETAIL_INDENT}  ... and {remaining} more")


# ---------------------------------------------------------------------------
# Diagnostics: deliberately outside the column
# ---------------------------------------------------------------------------


def _diagnostic(
    label: str, style: str, text: str, *, stderr: bool = False, indent: str = ""
) -> None:
    line = Text(f"{indent}{label}: {text}")
    _MESSAGE_HIGHLIGHTER.highlight(line)
    line.stylize(style, len(indent), len(indent) + len(label) + 1)
    (error_console if stderr else console).print(line, soft_wrap=True)


def note(text: str, *, stderr: bool = False) -> None:
    """`warning: ...` - something is off and the command carries on.

    Column zero, beside the operations: a problem has to break the left edge
    to be seen, and must not scan as one more step that went fine.
    Lowercase, as uv and cargo print it.

    `stderr` is for a command whose stdout is a *payload* rather than a
    report: `config show` emits valid TOML (see `toml` below), so a warning
    printed among it would end up in whatever file the user redirected it
    to. Everywhere else stdout is the report and a diagnostic belongs in it.
    """
    if _quiet:
        return
    _diagnostic("warning", "warning", text, stderr=stderr)


def failure(text: str) -> None:
    """`error: ...` - printed even under --quiet, which suppresses reports
    rather than problems."""
    _diagnostic("error", "failure", text)


# Advice is indented under the operation it follows. `warning:` and `error:`
# keep the margin, because a problem has to break the left edge to be seen; a
# hint is subordinate to the line above it - the next thing to do about what
# just happened - and indenting says so.
_HINT_INDENT = " " * _CONTENT_INDENT


def hint(text: str, *, stderr: bool = False) -> None:
    """`hint: ...` - the command to run next.

    `stderr` as for `note`: it follows the warning it belongs to.
    """
    if _quiet:
        return
    _diagnostic("hint", "boepie.hint", text, stderr=stderr, indent=_HINT_INDENT)


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


# An ASCII spinner, because the rest of boepie's own output is ASCII and a
# runtime-only glyph would be the one place that is not. rich ships several
# unicode ones; this is the classic four-frame line.
_SPINNER_FRAMES = ("-", "\\", "|", "/")
_SPINNER_SECONDS_PER_FRAME = 0.12

# The bar hangs under its label, not beside it. A description and a bar on one
# row is as wide as both, so on an ordinary terminal the counts and times get
# squeezed off the end - and the label is the part that says what is being
# waited for. Two lines cost nothing: the whole block is transient and is
# replaced by the operation line that summarises it.
_PROGRESS_INDENT = " " * _CONTENT_INDENT


class _StackedProgress(Progress):
    """rich's Progress with the task description on a line of its own.

    The description column is deliberately absent from the columns passed in;
    it is rendered here instead, behind a spinner, with the bar row beneath.
    """

    def get_renderables(self):
        frame = _SPINNER_FRAMES[
            int(time.monotonic() / _SPINNER_SECONDS_PER_FRAME) % len(_SPINNER_FRAMES)
        ]
        for task in self.tasks:
            label = Text(f"{frame} ", style="boepie.spinner")
            label.append(task.description, style="heading.line")
            yield label
        # Padding rather than table.padding, which is per-cell and so would
        # indent every column instead of the row.
        yield Padding(self.make_tasks_table(self.tasks), (0, 0, 0, len(_PROGRESS_INDENT)))


@contextlib.contextmanager
def progress_bar(description: str, total: int | None) -> Iterator[ProgressUpdate]:
    """A live bar for one long step, yielding the callable that advances it.

    Called with no arguments it advances by one; called with `(done, total)`
    it sets both, which is what a step whose total is not known up front
    needs - `build()` cannot say how many chunks there are until it has
    finished chunking, and renders as an indeterminate spinner until it can.

    **On stderr, and transient.** The bar is progress, not output: it is
    erased when the step ends, leaving the `operation` line that summarises
    it, and it never lands in a redirected file. `boepie setup > log.txt`
    therefore gives clean report lines with no bar residue - which is not
    true of a bar written to stdout, where every redraw is another set of
    escape codes in the file.

    Yields a no-op when `progress_wanted()` is False, so a caller never has
    to branch on it.
    """
    if not progress_wanted():
        yield lambda completed=None, total=None: None
        return

    with _StackedProgress(
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_progress_console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(description, total=total)

        def update(completed: int | None = None, total: int | None = None) -> None:
            if completed is None and total is None:
                progress.advance(task_id)
                return
            progress.update(task_id, completed=completed, total=total)

        yield update


# ---------------------------------------------------------------------------
# Verbosity
# ---------------------------------------------------------------------------

# --quiet suppresses the report but never a failure; --no-progress suppresses
# live bars but never the summary line that follows one. Module-level because
# they are set once from the CLI's own options and read by every primitive.
_quiet = False
_progress_wanted = True


def set_verbosity(*, quiet: bool = False, progress: bool = True) -> None:
    """Apply `--quiet` / `--no-progress` for the rest of the process."""
    global _quiet, _progress_wanted
    _quiet = quiet
    _progress_wanted = progress


def progress_wanted() -> bool:
    """Whether a live bar should be drawn. False under --no-progress, under
    --quiet, and whenever stdout is not a terminal - a bar redrawing itself
    into a redirected file is noise nobody asked for."""
    return _progress_wanted and not _quiet and console.is_terminal


# Set only by `following_steps(False)`. A composite command runs the steps
# its parts would otherwise be advising, and a sub-command has no way to
# know it is not the outermost caller.
_next_steps_wanted = True


@contextlib.contextmanager
def following_steps(wanted: bool) -> Iterator[None]:
    """Suppress the `Next:` advice inside the block when `wanted` is False.

    `corpus sync` closes by telling you to run `corpus index`. Inside
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
    command: str, *, before: str = "", note: str = "", stderr: bool = False
) -> None:
    """The "what to run now" line a command closes with, as a `hint:`.

    One helper so the phrasing and the command's styling stay identical
    everywhere; six call sites used to spell it out by hand. It reads as a
    diagnostic rather than an operation because that is what it is - advice
    about a command that has not run yet, which must not scan as a step that
    already did.
    """
    if not _next_steps_wanted or _quiet:
        return
    preamble = f"{before} " if before else ""
    trailer = f" {note}" if note else ""
    hint(f"{preamble}run `{command}`{trailer}".strip(), stderr=stderr)


# ---------------------------------------------------------------------------
# Fragments inside a message
# ---------------------------------------------------------------------------

# These return text rather than printing it: a command or a value is almost
# always part of a sentence a caller is composing, and the sentence belongs to
# the caller. What belongs here is the *delimiter*, because the delimiter is
# what the highlighter reads to decide the style - so a call site that picks
# its own is choosing a colour without knowing it. That is how boepie ended up
# suggesting `boepie context init` in cyan and 'boepie corpus list' in bold
# white in two errors a user could hit one after the other.


def command(invocation: str) -> str:
    """An invocation the reader is being told to run.

    Backticks, which is what uv, cargo and click all use and what survives
    being copied out of a terminal. Use it for any command - boepie's own or
    not - and never for a value.
    """
    return f"`{invocation}`"


def value(text: object) -> str:
    """Something the user typed or boepie read back: a config key, an id, a
    citekey, a filename. Quoted, so it never reads as a command."""
    return f"'{text}'"


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


def hint_coordinate(coordinate: str, snippet: str) -> None:
    """One `path#section: snippet` coordinate line from `boepie hint`.

    Named for the command rather than for the grammar: this is a payload
    `boepie hint` emits for a prompt hook, not the `hint:` diagnostic that
    suggests a command to run next.
    """
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


class PlainMessage(click.ClickException):
    """Base for every abort whose message boepie wrote itself.

    It exists so `cli._PlainErrorFormatter` has one type to check. These
    messages routinely end by naming the command that fixes the problem, and
    rich-click's bordered panel word-wraps that across a border into
    something the reader cannot copy - so they render through their own
    `show` instead. A new abort type that forgot to be listed there would
    silently get the panel back; subclassing this cannot.
    """


class Cancelled(PlainMessage):
    """The user stopped the command. Not a failure, and not a success.

    Its own type, because a composite has to be able to tell cancellation
    apart from the failures it deliberately carries on past. `sync` and
    `setup` warn-and-continue when a corpus fetch fails, so that whatever was
    already fetched still gets indexed - and cancellation used to be signalled
    as `SystemExit(130)`, which that wrapper caught as exactly such a failure.
    A Ctrl-C during the fetch was reported as `corpus fetch failed: 130`, every
    later phase ran anyway, and the command exited 0.

    Exit code 130 is the shell's convention for a process killed by SIGINT
    (128 + 2), which is what a caller checking `$?` in a loop expects.
    """

    exit_code = 130

    def show(self, file: object = None) -> None:
        # Not `Error:` - nothing went wrong, and calling it an error sends
        # the reader looking for a cause.
        line = Text()
        line.append("Cancelled: ", style="warning")
        message = Text(self.format_message())
        _MESSAGE_HIGHLIGHTER.highlight(message)
        line.append_text(message)
        error_console.print(line, soft_wrap=True)


class CliError(PlainMessage):
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
