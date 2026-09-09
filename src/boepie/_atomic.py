"""Writes that an interrupt cannot leave half-done.

Two shapes, because boepie's writers come in two kinds and the wrong one
does real damage:

``replace_file`` is for a file that replaces an existing one - a manifest, a
converged bundle file, a corpus document. The new bytes land in a sibling
temporary file and are renamed over the target, so a reader sees either the
old content or the new one and never a truncation. Each call stands alone,
which is what keeps a partially-finished run *resumable*: fifteen papers
fetched before a Ctrl-C are fifteen whole papers, and re-running the command
carries on from there.

``replacing_directory`` is for an artefact that is only meaningful complete -
a built index. The whole directory is staged beside the target and swapped in
at the end, so the previous one keeps serving for the entire build and an
interrupt costs the work but never the index that was already there. Applying
*this* shape to a corpus fetch would be a regression rather than a fix: it
would throw away every document written before the interrupt.

Both live at the package root because ``rag``, ``corpus``, ``context`` and
``mcp_config`` all write, and ``rag`` already imports ``corpus`` - a helper
in any one of them that the others reached back for would close that loop
(the same reason ``_glob`` sits here).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Staging names are dot-prefixed and carry the target's own name, so a sweep
# can recognise them without a registry and no real corpus or index entry can
# ever collide with one. The loaders skip dot-prefixed paths already.
_STAGING_PREFIX = "."
_STAGING_INFIX = ".staging-"
_RETIRED_INFIX = ".retiring-"


def replace_file(path: Path, data: str | bytes, *, encoding: str = "utf-8") -> None:
    """Write `data` to `path` atomically, creating parent directories.

    The target is left untouched if anything goes wrong, including a Ctrl-C
    partway through the write - hence `BaseException` rather than `Exception`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        if isinstance(data, bytes):
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
        else:
            with os.fdopen(handle, "w", encoding=encoding) as stream:
                stream.write(data)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@contextmanager
def replacing_directory(path: Path) -> Iterator[Path]:
    """Yield a staging directory whose contents replace `path` on clean exit.

    The caller writes everything into the yielded path. Only when the block
    finishes without raising is the staging directory swapped in - previous
    contents renamed aside, staging renamed into place, the old one deleted.
    Two renames rather than one, because `os.replace` will not overwrite a
    non-empty directory; the window between them is microseconds against the
    minutes an index build takes, which is the whole point.

    An interrupt inside the block deletes the staging directory and leaves
    `path` exactly as it was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _sweep_staging(path)
    staging = Path(
        tempfile.mkdtemp(dir=path.parent, prefix=f"{_STAGING_PREFIX}{path.name}{_STAGING_INFIX}")
    )
    try:
        yield staging
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    retired: Path | None = None
    if path.exists():
        retired = Path(
            tempfile.mkdtemp(
                dir=path.parent, prefix=f"{_STAGING_PREFIX}{path.name}{_RETIRED_INFIX}"
            )
        )
        # mkdtemp made it, so it exists and `os.replace` onto a directory
        # requires that directory to be empty - which it is, until now.
        retired.rmdir()
        os.replace(path, retired)
    try:
        os.replace(staging, path)
    except BaseException:
        if retired is not None:
            os.replace(retired, path)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if retired is not None:
        shutil.rmtree(retired, ignore_errors=True)


def _sweep_staging(path: Path) -> None:
    """Remove staging directories a previous run could not clean up itself.

    `replacing_directory` deletes its own on any exception, so this only ever
    finds one left by a kill that ran no handlers at all (SIGKILL, power
    loss). Left alone they would be a full copy of an index sitting on disk
    forever, so the next build of the same target clears them.
    """
    for candidate in path.parent.glob(f"{_STAGING_PREFIX}{path.name}.*"):
        name = candidate.name
        if _STAGING_INFIX in name or _RETIRED_INFIX in name:
            shutil.rmtree(candidate, ignore_errors=True)
