"""`boepie._display`: the shape of the CLI's report.

The verb column is load-bearing rather than decorative - an in-flight verb
and its finished form have to end in the same place for a progress line to be
*replaced* by its summary instead of scrolling away from it. These tests pin
the column, the three line kinds that hang off it, and the two switches that
turn output down.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from boepie import _display as display


@pytest.fixture(autouse=True)
def _default_verbosity() -> None:
    """Verbosity is module-level state set once from the CLI, so a test that
    changes it would otherwise leak into every test after it."""
    display.set_verbosity(quiet=False, progress=True)
    yield
    display.set_verbosity(quiet=False, progress=True)


def _rendered(function, *args, **kwargs) -> str:
    """What `function` prints, with styling stripped."""
    console = Console(record=True, width=200, force_terminal=False)
    original = display.console
    display.console = console
    try:
        function(*args, **kwargs)
    finally:
        display.console = original
    return console.export_text()


# ---------------------------------------------------------------------------
# the column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verb",
    ["Initialized", "Registering", "Indexed", "Built", "Reticulating"],
)
def test_every_verb_starts_at_the_margin(verb: str) -> None:
    """Indentation carries the structure, not alignment. Verbs used to be
    right-aligned into an 11-character column, uv's convention - which gave
    the payload one column at the cost of two left edges, since a detail
    marker cannot join that column without being inset ten characters.

    `Reticulating` is in the list on purpose: it is longer than every verb
    boepie uses and the old shape *refused* it, because one overflowing verb
    would have pushed past the column and broken the left edge for every
    other line. Nothing is aligned to a width now, so nothing can overflow
    one."""
    line = _rendered(display.operation, verb, "something").rstrip("\n")

    assert line == f"{verb} something"


def test_an_elapsed_time_is_appended_not_interpolated() -> None:
    line = _rendered(display.operation, "Indexed", "122 chunks", elapsed=1.4)

    assert line.strip() == "Indexed 122 chunks in 1.4s"


# ---------------------------------------------------------------------------
# elapsed formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.007, "7ms"),
        (0.039, "39ms"),
        (0.9, "900ms"),
        (1.4, "1.4s"),
        (44.5, "44.5s"),
        (145.0, "2m25s"),
        (3661.0, "61m01s"),
    ],
)
def test_elapsed_reads_at_the_precision_a_reader_can_act_on(
    seconds: float, expected: str
) -> None:
    """Milliseconds below a second, as uv prints them: `in 0.0s` on 40ms of
    work is a rounding artefact that tells the reader nothing."""
    assert display._format_elapsed(seconds) == expected


# ---------------------------------------------------------------------------
# details and diagnostics
# ---------------------------------------------------------------------------


def test_details_sit_one_step_under_the_operation_they_belong_to() -> None:
    line = _rendered(display.detail, "+", ".mcp.json").rstrip("\n")

    assert line == f"{' ' * display._CONTENT_INDENT}+ .mcp.json"


def test_a_long_detail_list_is_capped_with_a_count_of_the_rest() -> None:
    """A first docs fetch touches 98 pages; 98 titles is a wall nobody reads,
    and the count on the operation line above already said how many."""
    rendered = _rendered(display.details, "+", [f"paper{n}" for n in range(14)])

    assert rendered.count("+ ") == display.DETAIL_LIMIT
    assert "... and 4 more" in rendered


def test_verbose_lifts_the_cap() -> None:
    rendered = _rendered(
        display.details, "+", [f"paper{n}" for n in range(14)], limit=None
    )

    assert rendered.count("+ ") == 14
    assert "more" not in rendered


@pytest.mark.parametrize(
    ("function", "label"),
    [(display.note, "warning"), (display.failure, "error")],
)
def test_a_problem_starts_at_column_zero(function, label: str) -> None:
    """Deliberately outside the verb column: something went wrong, and it has
    to break the left edge rather than scan as one more step that went fine."""
    line = _rendered(function, "something happened").rstrip("\n")

    assert line == f"{label}: something happened"


def test_advice_hangs_under_the_operation_it_follows() -> None:
    """A hint is not a problem - it is the next thing to do about what just
    happened - so it sits under the operation text rather than out at the
    margin with the warnings, and clear of the detail markers at column 1."""
    line = _rendered(display.hint, "run something").rstrip("\n")

    assert line == f"{' ' * display._CONTENT_INDENT}hint: run something"


def test_next_step_is_a_hint_about_a_command_that_has_not_run() -> None:
    line = _rendered(display.next_step, "boepie corpus index").rstrip("\n")

    assert line.strip() == "hint: run `boepie corpus index`"


# ---------------------------------------------------------------------------
# verbosity
# ---------------------------------------------------------------------------


def test_quiet_suppresses_the_report() -> None:
    display.set_verbosity(quiet=True)

    assert _rendered(display.operation, "Indexed", "122 chunks") == ""
    assert _rendered(display.detail, "+", "a file") == ""
    assert _rendered(display.note, "something is off") == ""


def test_quiet_never_suppresses_a_failure() -> None:
    """--quiet turns down the report, not the problems: a command that failed
    has to say so even when asked to be silent."""
    display.set_verbosity(quiet=True)

    assert "error: it broke" in _rendered(display.failure, "it broke")


def test_no_progress_suppresses_the_bar() -> None:
    display.set_verbosity(progress=False)

    assert not display.progress_wanted()


def test_quiet_also_suppresses_the_bar() -> None:
    """A bar is report, so anything that silences the report silences it."""
    display.set_verbosity(quiet=True)

    assert not display.progress_wanted()


def test_a_suppressed_bar_still_yields_something_callable() -> None:
    """So no caller has to branch on whether a bar is wanted."""
    display.set_verbosity(progress=False)

    with display.progress_bar("Indexing", total=10) as advance:
        advance()
        advance(5, 10)


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


def test_a_diagnostic_goes_to_stdout_by_default() -> None:
    """stdout is the report on every command whose output *is* a report,
    which is nearly all of them."""
    assert "warning: something" in _rendered(display.note, "something")


@pytest.mark.parametrize(
    "function", [display.note, display.hint, display.next_step]
)
def test_a_diagnostic_can_be_put_on_stderr_instead(function) -> None:
    """For a command whose stdout is a payload rather than a report:
    `config show` emits valid TOML, and a warning printed into it would land
    in whatever file the output was redirected to."""
    console = Console(record=True, width=200, force_terminal=False)
    original_out, original_err = display.console, display.error_console
    display.console, display.error_console = original_out, console
    try:
        function("something", stderr=True)
    finally:
        display.console, display.error_console = original_out, original_err

    assert "something" in console.export_text()


# ---------------------------------------------------------------------------
# commands and values
# ---------------------------------------------------------------------------


def _styles(text: str) -> dict[str, str]:
    """Each highlighted fragment of `text` mapped to the style it got."""
    rendered = Text(text)
    display._MESSAGE_HIGHLIGHTER.highlight(rendered)
    return {text[span.start : span.end]: str(span.style) for span in rendered.spans}


def test_every_command_gets_one_style_whatever_it_names() -> None:
    """The bug this pins: the highlighter used to recognise an invocation by
    matching a hardcoded list of boepie's own subcommands, and `setup` was
    added to the CLI without being added to the list. `boepie setup` came out
    in the plain quoted style while `boepie context init` beside it came out
    cyan - two suggestions in two colours, for no reason a reader could see.
    """
    for invocation in (
        "boepie setup",
        "boepie context init",
        "boepie corpus sync --collection literature",
        "uv run scripts/migrate_corpus_layout.py",
    ):
        styles = _styles(f"Run {display.command(invocation)} first.")

        assert styles[f"`{invocation}`"] == "boepie.command"


def test_a_command_stays_one_span_over_the_options_inside_it() -> None:
    """`--collection` restyled on top of the invocation it belongs to lost the
    command's weight and made a single suggestion look like two."""
    rendered = Text(f"run {display.command('boepie corpus sync --collection docs')}")
    display._MESSAGE_HIGHLIGHTER.highlight(rendered)

    # Last span wins in a RegexHighlighter, so the command must come after the
    # option it contains.
    assert str(rendered.spans[-1].style) == "boepie.command"


def test_a_value_is_styled_apart_from_a_command() -> None:
    """A config key and the command that fixes it appear in one sentence
    constantly; telling them apart is the whole point of having two."""
    styles = _styles(
        f"unknown config key {display.value('foo.bar')}. "
        f"Run {display.command('boepie config show')}."
    )

    assert styles["'foo.bar'"] == "boepie.quoted"
    assert styles["`boepie config show`"] == "boepie.command"


def test_prose_that_merely_names_boepie_is_not_styled_as_a_command() -> None:
    """The reason the old pattern needed a subcommand list at all. A marker
    gets this right without one."""
    assert _styles("boepie is running on built-in defaults") == {}


def test_no_message_in_src_wraps_a_command_in_quotes() -> None:
    """The convention `_display` enforces is only worth having if nothing
    writes past it, and the failure is invisible: a quoted invocation still
    prints, just in a different colour from the backticked one in the next
    error, so only a reader hitting both would notice. Six messages had
    drifted this way before the delimiters moved into `command()`/`value()`.

    Every *string constant* is checked rather than every line, so a bare
    invocation handed to `next_step` or `command` - which apply the
    delimiter themselves - is not mistaken for one that wrote its own.
    """
    source_root = Path(display.__file__).parent
    quoted_command = re.compile(r"""['"](?:boepie|uv run|python -m) [^'"\n]*['"]""")

    def literal_text(node: ast.AST) -> str:
        """A string node's own text, with each interpolation as a placeholder.

        An f-string is where this drifts in practice - the delimiters sit in
        different Constant parts either side of a `{...}`, so checking the
        parts one at a time sees no quoted command at all and four of the six
        real cases went unnoticed.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(
                part.value
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
                else "{}"
                for part in node.values
            )
        return ""

    offenders: list[str] = []
    for module in sorted(source_root.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        nodes = list(ast.walk(tree))
        # A JoinedStr's own parts are walked too; they are covered by the
        # reconstruction above and would otherwise be checked in halves.
        inside_fstring = {
            id(part)
            for node in nodes
            if isinstance(node, ast.JoinedStr)
            for part in node.values
        }
        for node in nodes:
            if id(node) in inside_fstring:
                continue
            found = quoted_command.search(literal_text(node))
            if found:
                offenders.append(
                    f"{module.relative_to(source_root)}:{node.lineno}: {found.group()}"
                )

    assert not offenders, (
        "wrap a command with display.command(), which owns the delimiter the "
        "highlighter reads:\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# detail markers
# ---------------------------------------------------------------------------


def _styled(function, *args) -> list[tuple[str, str]]:
    """(fragment, style) for everything `function` styles."""
    console = Console(record=True, width=200, force_terminal=False)
    original = display.console
    display.console = console
    captured: list[Text] = []
    # Intercept rather than export: the styles are the point here, and
    # `export_text` throws them away.
    console.print = lambda text, **_: captured.append(text)
    try:
        function(*args)
    finally:
        display.console = original
    line = captured[0]
    return [(str(line)[s.start : s.end], str(s.style)) for s in line.spans]


@pytest.mark.parametrize(
    ("marker", "style"),
    [("+", "boepie.added"), ("-", "boepie.removed"), ("~", "boepie.changed")],
)
def test_the_marker_carries_the_colour(marker: str, style: str) -> None:
    """What a reader scans a list of details for is which kind each one is,
    and that is one character in the same column every time - so the glyph is
    coloured, as uv colours it, and the name beside it stays dim."""
    assert (marker, style) in _styled(display.detail, marker, "a-file")


def test_an_unchanged_marker_gets_no_colour_of_its_own() -> None:
    """`=` is the absence of news."""
    styles = dict(_styled(display.detail, "=", "a-file"))

    assert styles.get("=") == "boepie.marker"


# ---------------------------------------------------------------------------
# the progress spinner
# ---------------------------------------------------------------------------


def test_every_spinner_frame_is_ascii() -> None:
    """rich ships several spinners and all of them are unicode. boepie's own
    output is ASCII throughout, and a runtime-only glyph would be the one
    place it is not."""
    for frame in display._SPINNER_FRAMES:
        assert frame.isascii()


# ---------------------------------------------------------------------------
# one rule for the severity colour
# ---------------------------------------------------------------------------


def test_every_operation_style_is_one_of_the_three() -> None:
    """Colour tracks what happened, not which noun it happened to: green for
    work done, dim for a no-op, yellow for something that needs attention.
    A reader scans the verbs for yellow, so a fourth meaning would cost that.

    Checked as a property of the call sites because the failure is invisible
    - a wrong style still prints, just in a colour that means something else,
    and only a reader comparing two runs would ever notice.
    """
    import ast

    tree = ast.parse(
        (Path(display.__file__).parent / "cli.py").read_text(encoding="utf-8")
    )
    styles: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "operation"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "style" and isinstance(keyword.value, ast.Constant):
                styles.add(keyword.value.value)

    assert styles <= {"success", "muted", "warning"}, styles
