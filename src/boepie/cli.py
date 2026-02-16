"""Command-line interface for the boepie application.

This module provides the CLI commands and configuration validation for boepie,
including IP address parsing and server configuration management.
"""

from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Annotated, Any, Literal

import click
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    TypeAdapter,
)

from boepie import __version__
from boepie.app import mcp
from boepie.utilities import full_class_name

_LOCALHOST_ADDRESS = IPv4Address("127.0.0.1")


def validate_ip_address(value: Any) -> IPv4Address | IPv6Address:
    """Validate and convert a value to an IP address object.

    Accepts IP address strings, IP address objects, or the keywords
    'default' and 'localhost' which resolve to 127.0.0.1.
    """
    # Handle string input
    if isinstance(value, str):
        stripped = value.strip().lower()
        # Handle keywords
        if stripped in ("default", "localhost"):
            return _LOCALHOST_ADDRESS
        # Parse as IP address
        try:
            return ip_address(stripped)
        except ValueError as error:
            raise ValueError(f"Invalid IP address string {value!r}") from error
    # Pass through already-validated IP address objects
    elif isinstance(value, (IPv4Address, IPv6Address)):
        return value
    # Reject any other type
    else:
        types = ", ".join(
            (
                full_class_name(str),
                full_class_name(IPv4Address),
                full_class_name(IPv6Address),
            )
        )
        raise ValueError(f"Expected {types}, got {full_class_name(value)}")


# Type alias for IP addresses with automatic validation
# This can be used in Pydantic models to ensure IP address fields are validated
IPAddress = Annotated[IPv4Address | IPv6Address, BeforeValidator(validate_ip_address)]

# Type adapter for standalone IP address validation outside of Pydantic models
ipaddress_adapter: TypeAdapter[IPv4Address | IPv6Address] = TypeAdapter(IPAddress)


def parse_ip_address(
    address: str | IPv4Address | IPv6Address,
) -> IPv4Address | IPv6Address:
    """Parse and validate an IP address.

    Handles string representations, existing IP address objects, and keywords
    'default' and 'localhost' (which resolve to 127.0.0.1).
    """
    return ipaddress_adapter.validate_python(address)


class ServerConfig(BaseModel):
    """Configuration model for the boepie server.

    This Pydantic model validates and stores server configuration parameters,
    ensuring that the host is a valid IP address and the port is within the
    valid range for TCP/UDP ports.

    Attributes:
        host: The IP address to bind the server to. Can be IPv4 or IPv6.
        port: The TCP port number to bind the server to. Must be between
            1 and 65535 (inclusive). Defaults to 8000.

    Examples:
        >>> config = ServerConfig(host="127.0.0.1", port=8080)
        >>> config.host
        IPv4Address('127.0.0.1')
        >>> config.port
        8080
    """

    transport: Literal["stdio", "http", "sse", "streamable-http"]
    "Validated transport protocol to use"

    host: IPAddress
    "Validated IP address (IPv4 or IPv6)"

    port: int = Field(default=8000, ge=1, le=65535)
    "Valid port (in range 1 to 65535)"


@click.group()
@click.version_option(version=__version__, prog_name="boepie")
def cli():
    """Main entry point for the boepie command-line interface.

    This is the root command group that organizes all boepie CLI subcommands.
    Use 'boepie --help' to see available commands, or 'boepie --version' to
    display the current version.
    """
    pass


@cli.command()
@click.option(
    "--transport",
    default="stdio",
    show_default=True,
    type=click.Choice(("stdio", "http", "sse", "streamable-http")),
    help="Transport protocol (medium) to use for the server.",
)
@click.option(
    "--host", default="127.0.0.1", show_default=True, help="Host to bind the server to."
)
@click.option(
    "--port",
    default=8000,
    show_default=True,
    type=int,
    help="Port to bind the server to.",
)
def serve(transport: Any, host: Any, port: Any):
    """Start the boepie server.

    This command starts the boepie application server with the specified
    host and port configuration. The server configuration is validated
    before starting.

    """
    # Validate configuration using Pydantic model
    config = ServerConfig(transport=transport, host=host, port=port)
    click.echo(f"boepie serve called with host={config.host} port={config.port}")

    if config.transport == "http":
        mcp.run("http", host=config.host, port=config.port)
    else:
        mcp.run(config.transport)


@cli.command()
@click.option(
    "--force", is_flag=True, help="Force initialization, overwriting existing files."
)
def init(force):
    """Initialize a new boepie project.

    This command sets up a new boepie project in the current directory by
    creating necessary configuration files and directory structure.
    """
    # TODO: Implement project initialization logic
    click.echo(f"boepie init called force={force}")
