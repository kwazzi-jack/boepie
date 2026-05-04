"""Command-line interface for boepie."""

from __future__ import annotations

import click

from boepie import __version__
from boepie.server import mcp


@click.group()
@click.version_option(version=__version__, prog_name="boepie")
def cli() -> None:
    """Boepie - MCP server for AI-assisted stimela pipeline creation."""


@cli.command()
def serve() -> None:
    """Start the boepie MCP server (stdio transport)."""
    mcp.run("stdio")
