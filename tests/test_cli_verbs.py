"""The verb model: one word means one thing, at every noun.

boepie's commands used to name the same operation three ways - `context
apply`, `corpus fetch`, `index build` were all "make this match its source"
- so a user had to memorise which noun took which verb. They are all `sync`
now, and the scaffold half is `init` everywhere it exists.

These tests pin the *shape* rather than any one command, because the value of
the model is entirely in its uniformity: a fifth noun added with a fourth
word for converging would cost more than the word.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from boepie import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _groups() -> dict[str, click.Group]:
    return {
        name: command
        for name, command in cli.cli.commands.items()
        if isinstance(command, click.Group)
    }


def test_every_noun_that_scaffolds_calls_it_init() -> None:
    """`config create` was an `init` under a different name - it makes a file
    that is not there yet. `index` legitimately has none: an index is wholly
    derived from a corpus, so there is nothing to scaffold that `sync` would
    not recreate."""
    scaffolds = {name for name, group in _groups().items() if "init" in group.commands}

    assert scaffolds == {"context", "corpus", "config"}
    assert "create" not in _groups()["config"].commands


def test_every_noun_that_converges_calls_it_sync() -> None:
    """`config` legitimately has none: there is no upstream to converge with,
    the file is the source."""
    converges = {name for name, group in _groups().items() if "sync" in group.commands}

    assert converges == {"context", "corpus"}
    for retired, group in (("apply", "context"), ("fetch", "corpus")):
        assert retired not in _groups()[group].commands


def test_indexing_belongs_to_the_noun_that_owns_the_index() -> None:
    """There is no `index` noun. The corpus collections index to the
    machine-global `INDEX_DIR/<collection>/` and the bundle indexes to
    `<bundle>/.index/` inside the project - no shared storage, no shared
    scope, no shared retrieval stack. The old grouping proved itself wrong:
    `index status` enumerated INDEX_DIR alone, so the noun that claimed to
    own every index could not see one of them.

    It also cost a duplicated selector - `corpus sync -l` then `index sync
    --collection literature` named the same thing twice, in two spellings.
    """
    assert "index" not in _groups()
    assert "index" in _groups()["corpus"].commands
    assert "index" in _groups()["context"].commands


def test_index_is_the_composite_for_the_two_indexing_verbs() -> None:
    """As `boepie sync` is for the two converging ones. It fills the gap sync
    leaves: rebuilding every index *without* fetching, which is what a change
    of embedding backend or model needs - the corpus has not moved, but every
    vector in it has to be recomputed."""
    assert "index" in cli.cli.commands
    assert not isinstance(cli.cli.commands["index"], click.Group)


def test_the_top_level_verbs_are_init_sync_register_and_setup() -> None:
    """`setup` is exactly the other three in order and adds nothing of its
    own, so each can be had without the others: `init` touches no network,
    `sync` changes no registration, `register` writes no content."""
    for name in ("init", "sync", "register", "setup"):
        assert name in cli.cli.commands


def test_setup_composes_the_three_and_adds_nothing() -> None:
    """Its options are exactly the union of what it delegates, which is what
    makes it an alias rather than a fourth thing to keep in step."""
    composed = set()
    for name in ("init", "sync", "register"):
        composed |= {option.name for option in cli.cli.commands[name].params}
    setup_options = {option.name for option in cli.cli.commands["setup"].params}

    # `--check` is the exception, and deliberately: it makes `register`
    # read-only, which is the opposite of what a composite that scaffolds,
    # fetches and writes is for.
    assert setup_options == composed - {"check_only"}


def test_sync_takes_no_only_option() -> None:
    """`--only context` was a second spelling of `boepie context sync`, and a
    second place for the two to disagree."""
    assert "only" not in {option.name for option in cli.cli.commands["sync"].params}


@pytest.mark.parametrize("group", ["context", "corpus"])
def test_every_noun_reports_state_the_same_way(group: str) -> None:
    """`status` is the read verb throughout, and no noun answers it with a
    different word."""
    assert "status" in _groups()[group].commands


def test_registration_is_not_a_phase_of_init() -> None:
    """The one genuinely optional part of setting a workspace up - you may
    want the bundle and the corpora without changing any agent's config - so
    it is its own command rather than something `init` does to you."""
    init_options = {option.name for option in cli.cli.commands["init"].params}

    assert "agents" not in init_options
    assert "agents" in {option.name for option in cli.cli.commands["register"].params}
