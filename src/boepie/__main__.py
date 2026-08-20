# boepie/__main__.py
"""Console-script entry point.

`boepie.cli` is imported inside `main`, not at module level, so that a
`ConfigError` can be reported as the actionable message it already carries
instead of a pydantic traceback. `boepie.config` resolves and validates
every setting at import time, and `boepie.cli` imports it - so without this
guard a single typo in a hand-edited `config.toml` makes every boepie
command dump a stack trace, including the `boepie config` commands that
would fix it.
"""

from __future__ import annotations

import sys


def main() -> None:
    from boepie.settings import ConfigError

    try:
        from boepie.cli import cli

        cli()
    except ConfigError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
