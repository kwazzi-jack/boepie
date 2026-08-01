# /// script
# requires-python = ">=3.12"
# dependencies = ["click", "rich"]
# ///

"""
package_content.py
-------------------
Package a curated knowledge-bundle content directory (index.md, log.md,
concepts/, playbooks/, cabs/, literature/) into the `knowledge-content.tar.gz`
release asset that `boepie knowledge fetch` expects: a tarball whose root
holds the content files directly (no wrapping directory, unlike
package_index.py's per-index-id layout - there is only ever one content
cache) plus a `content-manifest.json` recording `content_version` and
`generated_at`, and a `.sha256` sidecar with the tarball's checksum.

Usage:
    uv run scripts/package_content.py content/ --content-version 0.2.0 --out dist/
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import click
from rich.console import Console

console = Console()

_ASSET_NAME = "knowledge-content.tar.gz"
_MANIFEST_FILENAME = "content-manifest.json"


@click.command()
@click.argument("content_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--content-version", required=True, help="Version tag for this content snapshot (e.g. '0.2.0').")
@click.option("--out", default="dist", show_default=True, type=click.Path(path_type=Path), help="Output directory for the tarball + sidecar.")
def package_content(content_dir: Path, content_version: str, out: Path) -> None:
    """Tar CONTENT_DIR's entries (unwrapped) plus a content-manifest.json into a release asset."""
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / _ASSET_NAME
    checksum_path = out / f"{_ASSET_NAME}.sha256"

    # Record what actually went in, not just the version label: `apply` deletes
    # bundle files that are no longer in the content set, so "which files did
    # this snapshot contain" is the difference between a deliberate removal and
    # a packaging accident.
    files = sorted(
        path.relative_to(content_dir).as_posix()
        for path in content_dir.rglob("*")
        if path.is_file()
    )
    manifest_data = {
        "content_version": content_version,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_count": len(files),
        "files": files,
    }
    manifest_bytes = json.dumps(manifest_data, indent=2).encode("utf-8")

    with console.status(f"Packaging {content_dir} -> {tar_path}..."):
        with tarfile.open(tar_path, mode="w:gz") as tar:
            for entry in sorted(content_dir.iterdir()):
                tar.add(entry, arcname=entry.name)
            manifest_info = tarfile.TarInfo(name=_MANIFEST_FILENAME)
            manifest_info.size = len(manifest_bytes)
            tar.addfile(manifest_info, io.BytesIO(manifest_bytes))
        digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        checksum_path.write_text(f"{digest}  {_ASSET_NAME}\n", encoding="utf-8")

    size_mb = tar_path.stat().st_size / (1024 * 1024)
    console.print(f"[green]Packaged[/green] {_ASSET_NAME} ({size_mb:.1f} MiB) -> {tar_path}")
    console.print(f"[green]Checksum[/green] {digest} -> {checksum_path}")
    console.print(
        f"\nPublish with:\n"
        f"  gh release create <tag> {tar_path} {checksum_path} "
        f"--repo kwazzi-jack/boepie --title <title> --notes <notes>\n"
        f"or, to add to an existing release:\n"
        f"  gh release upload <tag> {tar_path} {checksum_path} --repo kwazzi-jack/boepie"
    )


if __name__ == "__main__":
    package_content()
