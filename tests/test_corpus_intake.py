"""Tests for boepie.corpus.intake: the shared format dispatch behind every
`corpus add` subcommand.

The behaviour that matters here is that no input produces a raw Python
traceback as its error message - the pre-refactor intake read every local
file as UTF-8, so a PDF surfaced a `UnicodeDecodeError` to the user.
"""

from __future__ import annotations

import subprocess

import pytest

from boepie.corpus.intake import (
    IntakeError,
    _mineru_environment,
    _mineru_failure_reason,
    convert_local_file,
    convert_html,
    detect_format,
    looks_like_url,
    read_text_file,
    sha256_of,
    title_from_markdown,
)


def _convert(path):
    return convert_local_file(
        path, keep_original=False, mineru_device_mode="auto",
        mineru_backend="pipeline", mineru_model_source="auto",
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("paper.pdf", "pdf"),
        ("slides.pptx", "pptx"),
        ("report.docx", "docx"),
        ("sheet.xlsx", "xlsx"),
        ("notes.md", "markdown"),
        ("notes.markdown", "markdown"),
        ("page.html", "html"),
        ("readme.txt", "text"),
        ("guide.rst", "text"),
        ("solver.py", "code"),
        ("config.yaml", "code"),
        ("Makefile", "code"),
    ],
)
def test_detect_format_classifies_by_suffix(tmp_path, name, expected):
    assert detect_format(tmp_path / name) == expected


def test_markdown_is_read_verbatim(tmp_path):
    source = tmp_path / "note.md"
    source.write_text("# Title\n\nBody.\n", encoding="utf-8")

    converted = _convert(source)

    assert converted.markdown == "# Title\n\nBody.\n"
    assert converted.via == "verbatim"
    assert converted.format == "markdown"
    assert converted.suggested_title == "Title"


def test_source_code_is_fenced_with_its_language(tmp_path):
    """Fenced so the chunker treats it as one block instead of reflowing it
    as prose, and so a reader can tell code from commentary."""
    source = tmp_path / "solver.py"
    source.write_text("def solve():\n    return 1\n", encoding="utf-8")

    converted = _convert(source)

    assert "```python" in converted.markdown
    assert "def solve():" in converted.markdown
    assert converted.format == "code"


def test_a_checksum_is_recorded_for_every_local_file(tmp_path):
    source = tmp_path / "note.md"
    source.write_bytes(b"# Title\n")

    converted = _convert(source)

    assert converted.sha256 == sha256_of(b"# Title\n")


def test_originals_are_only_retained_when_asked(tmp_path):
    source = tmp_path / "note.md"
    source.write_text("# Title\n", encoding="utf-8")

    assert _convert(source).original_bytes is None
    kept = convert_local_file(
        source, keep_original=True, mineru_device_mode="auto",
        mineru_backend="pipeline", mineru_model_source="auto",
    )
    assert kept.original_bytes == b"# Title\n"
    assert kept.original_name == "note.md"


def test_a_pdf_without_mineru_explains_how_to_install_it(tmp_path, monkeypatch):
    """Regression: this used to be `'utf-8' codec can't decode byte 0x8f`."""
    monkeypatch.setattr("boepie.corpus.intake.mineru_available", lambda: False)
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4\n\x8f\x8f binary\n")

    with pytest.raises(IntakeError) as error:
        _convert(source)

    message = str(error.value)
    assert "MinerU is required" in message
    assert "uv sync --extra mineru" in message
    assert "codec" not in message


def test_a_non_utf8_text_file_still_reads(tmp_path):
    """A text file written on a Windows box is not a reason to fail."""
    source = tmp_path / "note.txt"
    source.write_bytes("caf\xe9 latte\n".encode("cp1252"))

    assert "caf" in read_text_file(source)


def test_looks_like_url_only_matches_http_schemes():
    assert looks_like_url("https://example.org/page")
    assert looks_like_url("http://example.org/page")
    assert not looks_like_url("/home/someone/note.md")
    assert not looks_like_url("ftp://example.org/file")


def test_convert_html_strips_chrome():
    markdown = convert_html(
        "<html><body><nav>Menu</nav><script>x()</script>"
        "<h1>Real Title</h1><p>Real body.</p><footer>Legal</footer></body></html>"
    )

    assert "Real Title" in markdown
    assert "Real body." in markdown
    assert "Menu" not in markdown
    assert "Legal" not in markdown
    assert "x()" not in markdown


def test_title_from_markdown_falls_back_when_there_is_no_heading():
    assert title_from_markdown("Just prose.\n", "fallback") == "fallback"
    assert title_from_markdown("# Real\n\nProse.\n", "fallback") == "Real"


# ---------------------------------------------------------------------------
# MinerU invocation
# ---------------------------------------------------------------------------


def test_auto_settings_unset_the_variables_rather_than_passing_auto(monkeypatch):
    """MinerU expects auto-detection to be expressed by absence: it rejects
    the literal string with 'MINERU_MODEL_SOURCE=auto is not supported'."""
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "auto")
    monkeypatch.setenv("MINERU_DEVICE_MODE", "auto")

    environment = _mineru_environment(device_mode="auto", model_source="auto")

    assert "MINERU_MODEL_SOURCE" not in environment
    assert "MINERU_DEVICE_MODE" not in environment


def test_explicit_settings_are_passed_through(monkeypatch):
    monkeypatch.delenv("MINERU_MODEL_SOURCE", raising=False)

    environment = _mineru_environment(device_mode="cuda", model_source="huggingface")

    assert environment["MINERU_DEVICE_MODE"] == "cuda"
    assert environment["MINERU_MODEL_SOURCE"] == "huggingface"


def test_the_rest_of_the_environment_survives(monkeypatch):
    monkeypatch.setenv("SOME_UNRELATED_VAR", "keep-me")

    environment = _mineru_environment(device_mode="auto", model_source="auto")

    assert environment["SOME_UNRELATED_VAR"] == "keep-me"


def _failed(stdout: str = "", stderr: str = "", returncode: int = 1):
    return subprocess.CompletedProcess(
        args=["mineru"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_failure_reason_extracts_mineru_error_field_from_its_json_blob():
    """MinerU reports failures as a one-line JSON blob far too noisy to show
    verbatim; its `error` field is the part that says what went wrong."""
    blob = (
        '- task#1 (paper): Task 875022ca failed for task#1 : '
        '{"task_id": "875022ca", "status": "failed", "backend": "pipeline", '
        '"error": "MINERU_MODEL_SOURCE=auto is not supported.", "queued_ahead": 0}'
    )

    assert _mineru_failure_reason(_failed(stdout=blob)) == (
        "MINERU_MODEL_SOURCE=auto is not supported."
    )


def test_failure_reason_falls_back_to_the_last_output_line():
    assert _mineru_failure_reason(_failed(stderr="something broke\nfatal: no models")) == (
        "fatal: no models"
    )


def test_failure_reason_falls_back_to_the_exit_code_with_no_output():
    assert _mineru_failure_reason(_failed(returncode=137)) == "exit code 137"


# ---------------------------------------------------------------------------
# Title extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("# **Array Configuration Memo**", "Array Configuration Memo"),
        ("# *Italic Title*", "Italic Title"),
        ("# `code_title`", "code_title"),
        ("# [Linked Title](https://example.org)", "Linked Title"),
        ("# Plain Title", "Plain Title"),
        ("# **Mixed** and *more*", "Mixed and more"),
    ],
)
def test_a_title_never_carries_markdown_syntax(heading, expected):
    """MinerU renders a bold DOCX heading as `# **Title**`; those markers
    would otherwise reach the title field and from there every search hit,
    `corpus tree` line, and filename."""
    assert title_from_markdown(f"{heading}\n\nBody.\n", "fallback") == expected


def test_title_extraction_leaves_snake_case_identifiers_alone():
    """The stripping is narrow on purpose: underscores that are not a matched
    pair wrapping the whole word are ordinary characters."""
    assert title_from_markdown("# solver_py_notes\n", "fallback") == "solver_py_notes"


def test_an_emphasis_only_heading_falls_back():
    assert title_from_markdown("# ****\n\nBody.\n", "fallback") == "fallback"


def test_docx_derived_title_reaches_the_filename_cleanly(tmp_path):
    from boepie.corpus.layout import full_title_filename

    title = title_from_markdown("# **Array Configuration Memo**\n", "fallback")

    assert full_title_filename(title) == "Array Configuration Memo.md"
