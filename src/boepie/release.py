"""Shared GitHub-release-asset helpers: URL construction and checksum-verified download.

One consumer remains: `boepie.context.bundle`, which fetches the curated
`.boepie/` content tarball. Built search indices used to be published the
same way and are not any more - an index is now always built on the machine
that queries it, so `index fetch` and the assets behind it are gone.

Keeps the `.sha256` sidecar convention: every asset `<name>` is published
alongside a `<name>.sha256` file holding `<digest>  <name>`.
"""

from __future__ import annotations

import hashlib

import httpx

# Release assets live on the project's own public GitHub repo - no auth needed.
GITHUB_REPO = "kwazzi-jack/boepie"

_DOWNLOAD_TIMEOUT_SECONDS = 120


def release_asset_url(tag: str, asset_name: str) -> str:
    """Build the download URL for `asset_name` published under release `tag`.

    `tag="latest"` resolves GitHub's alias for the most recently published
    release rather than naming a specific tag.
    """
    if tag == "latest":
        return f"https://github.com/{GITHUB_REPO}/releases/latest/download/{asset_name}"
    return f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/{asset_name}"


def fetch_asset_checksum(tag: str, asset_name: str) -> str:
    """Fetch just `asset_name`'s `.sha256` sidecar and return the expected digest.

    One small request, letting a caller check whether a previously downloaded
    asset is still current before paying for the full asset download.
    """
    url = release_asset_url(tag, asset_name)
    with httpx.Client(follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as client:
        checksum_response = client.get(url + ".sha256")
        checksum_response.raise_for_status()
    return checksum_response.text.strip().split()[0]


def download_verified_asset(tag: str, asset_name: str, expected_digest: str | None = None) -> bytes:
    """Download `asset_name` from release `tag` and verify it against its `.sha256` sidecar.

    Pass `expected_digest` (e.g. already fetched via `fetch_asset_checksum`)
    to skip re-requesting the sidecar. Raises `ValueError` on a checksum
    mismatch (corrupt or tampered download); raises `httpx.HTTPStatusError`
    if the asset or sidecar is missing on the release.
    """
    url = release_asset_url(tag, asset_name)
    with httpx.Client(follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.content
        if expected_digest is None:
            checksum_response = client.get(url + ".sha256")
            checksum_response.raise_for_status()
            expected_digest = checksum_response.text.strip().split()[0]

    actual_digest = hashlib.sha256(data).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError(
            f"Checksum mismatch for {asset_name}: expected {expected_digest}, got "
            f"{actual_digest}. The download may be corrupt or tampered with."
        )
    return data
