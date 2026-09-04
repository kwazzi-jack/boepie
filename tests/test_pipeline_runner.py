"""`boepie.pipeline.runner._run`: bounding a child process boepie shells out to.

These calls sit behind MCP tools, which is what makes a hang and an uncaught
`TimeoutExpired` the same class of failure: neither reaches the agent as
something it can act on. So a timeout comes back as an ordinary failed
`RunResult`, and no child inherits a terminal it could block on.
"""

from __future__ import annotations

import subprocess

from boepie.pipeline import runner


def test_a_command_that_finishes_reports_its_own_exit_code() -> None:
    result = runner._run(["true"], timeout=30)

    assert result.ok
    assert result.returncode == 0


def test_a_timeout_comes_back_as_a_failed_result_not_an_exception() -> None:
    """A `TimeoutExpired` propagating out of an MCP tool is a stack trace the
    agent cannot act on; the same event as a non-zero result reads as
    "stimela did not finish", which is what happened."""
    result = runner._run(["sleep", "5"], timeout=1)

    assert not result.ok
    assert result.returncode == runner._TIMEOUT_RETURNCODE
    assert "Timed out after 1s" in result.stderr
    assert "cancelled" in result.stderr


def test_a_timeout_message_is_what_the_caller_renders() -> None:
    """`output` is what every recipe tool forwards on failure, so the reason
    has to be reachable from there rather than only from `stderr`."""
    result = runner._run(["sleep", "5"], timeout=1)

    assert "Timed out" in result.output


def test_a_timeout_keeps_whatever_the_child_managed_to_write() -> None:
    """The partial output is the only clue to where a run got stuck, and
    `TimeoutExpired` carries it as bytes rather than text."""
    result = runner._run(
        ["sh", "-c", "echo started work; sleep 5"], timeout=1
    )

    assert "started work" in result.stdout


def test_a_child_cannot_read_the_terminal(monkeypatch) -> None:
    """A child that decides to prompt must fail immediately rather than wait
    on a terminal that may not be there at all - under MCP there is nobody to
    answer it, and behind captured output a silent prompt is
    indistinguishable from a hang."""
    recorded: dict[str, object] = {}

    def _record(args, **kwargs):
        recorded.update(kwargs)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", _record)
    runner._run(["true"], timeout=30)

    assert recorded["stdin"] is subprocess.DEVNULL


def test_no_timeout_is_still_allowed() -> None:
    """`run_recipe` passes None deliberately: a real pipeline can take hours,
    and a cap boepie invented would kill it partway."""
    result = runner._run(["true"], timeout=None)

    assert result.ok
