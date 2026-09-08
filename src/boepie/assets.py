# boepie/assets.py
"""Packaged assets: the curated files boepie ships inside its own installation.

Three things travel in the wheel rather than over the network - the context
bundle's seed content, the arXiv paper manifest, and the documentation-site
manifest - and this module is the one place that knows where an installation
keeps them. A caller asks for the asset by name and gets an absolute path
into whichever venv boepie is imported from; nothing reconstructs a path from
`__file__` on its own any more.

**There is no other source, and no fallback.** boepie used to fetch the
context content as a `knowledge-content.tar.gz` GitHub release asset into a
machine-global cache, preferring that cache over the packaged seeds; the
cache could disagree with the installed boepie, and reconciling the two was
hidden behaviour with no way to see it from the outside. The venv is now the
whole answer: install boepie, and you have the content it was built with. An
asset missing from an installation is a broken install, not a case to
recover from, so it raises `MissingAssetError` naming what was looked for.

`asset_checksum` digests any of them - a file by its bytes, a directory by
its whole tree - so a caller that has to notice content changing under it
(the `.boepie/` bundle, which records the digest it was generated from) can
ask one question and get one answer.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path


class MissingAssetError(RuntimeError):
    """An asset that ships in the wheel is not present in this installation."""


def _packaged_path(package: str, name: str) -> Path:
    """Resolve `name` inside installed `package` as a real filesystem path.

    `importlib.resources.files` rather than `__file__` arithmetic so the
    answer comes from the import system that actually loaded the package -
    the venv boepie is running in, not whichever checkout happens to be
    beside the caller. boepie is installed unzipped (it ships a console
    script and reads its own data directories), so the traversable is always
    a real path; a zipimported install would fail the `exists()` check below
    rather than silently reading nothing.
    """
    resolved = Path(str(files(package).joinpath(name)))
    if not resolved.exists():
        raise MissingAssetError(
            f"{package}/{name} is missing from this boepie installation "
            f"(looked in {resolved}). Reinstall boepie."
        )
    return resolved


def context_content_dir() -> Path:
    """The curated seed content `context init`/`apply` copy a bundle from."""
    return _packaged_path("boepie.context", "content")


def literature_manifest_path() -> Path:
    """The arXiv papers `corpus sync --collection literature` reconciles against."""
    return _packaged_path("boepie.literature", "default_manifest.json")


def docs_manifest_path() -> Path:
    """The documentation sites `corpus sync --collection docs` reconciles against."""
    return _packaged_path("boepie.docs", "default_manifest.json")


def asset_checksum(asset_path: Path) -> str:
    """The sha256 of `asset_path`: its bytes for a file, its whole tree for a
    directory.

    A directory's digest covers each file's path *and* its bytes, walked in
    sorted order, so it is stable across filesystems and changes when a file
    is renamed, added or removed - not only when one is edited. That is what
    makes it usable as "did the content move under me": `apply` deletes
    bundle files whose source counterpart is gone, so a removal has to be
    visible.
    """
    if asset_path.is_file():
        return hashlib.sha256(asset_path.read_bytes()).hexdigest()

    digest = hashlib.sha256()
    for path in sorted(asset_path.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(asset_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
