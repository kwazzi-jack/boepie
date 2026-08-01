# /// script
# requires-python = ">=3.12"
# dependencies = ["click", "rich"]
# ///

"""
package_index.py
-----------------
Package a built boepie collection directory (as produced by `boepie index`)
into a release-ready tarball + sha256 sidecar, matching the layout
`boepie index fetch` expects: `{collection}-{index_id}.tar.gz` containing a
single top-level `{index_id}/` directory, plus a `.sha256` sidecar with the
tarball's checksum.

Usage:
    uv run scripts/package_index.py .index/literature/ollama-nomic-embed-text \\
        --collection literature --out dist/
"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.argument("collection_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--collection", required=True, help="Collection name (e.g. 'literature').")
@click.option("--out", default="dist", show_default=True, type=click.Path(path_type=Path), help="Output directory for the tarball + sidecar.")
def package_index(collection_dir: Path, collection: str, out: Path) -> None:
    """Tar COLLECTION_DIR (a built <index_id>/ directory) into a release asset."""
    index_id = collection_dir.name
    out.mkdir(parents=True, exist_ok=True)
    asset_name = f"{collection}-{index_id}.tar.gz"
    tar_path = out / asset_name
    checksum_path = out / f"{asset_name}.sha256"

    with console.status(f"Packaging {collection_dir} -> {tar_path}..."):
        with tarfile.open(tar_path, mode="w:gz") as tar:
            tar.add(collection_dir, arcname=index_id)
        digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        checksum_path.write_text(f"{digest}  {asset_name}\n", encoding="utf-8")

    size_mb = tar_path.stat().st_size / (1024 * 1024)
    console.print(f"[green]Packaged[/green] {asset_name} ({size_mb:.1f} MiB) -> {tar_path}")
    console.print(f"[green]Checksum[/green] {digest} -> {checksum_path}")
    console.print(
        f"\nPublish with:\n"
        f"  gh release create <tag> {tar_path} {checksum_path} "
        f"--repo kwazzi-jack/boepie --title <title> --notes <notes>\n"
        f"or, to add to an existing release:\n"
        f"  gh release upload <tag> {tar_path} {checksum_path} --repo kwazzi-jack/boepie"
    )


if __name__ == "__main__":
    package_index()
