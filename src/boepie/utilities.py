import csv
import io
from typing import Any


def class_name(obj: Any) -> str:
    if isinstance(obj, type):
        # obj is a class
        return obj.__name__
    else:
        # obj is an instance
        return type(obj).__name__


def full_class_name(obj: Any) -> str:
    """Get fully qualified class name including module path."""
    if isinstance(obj, type):
        # obj is a class
        cls = obj
    else:
        # obj is an instance
        cls = type(obj)

    module = cls.__module__
    qualname = cls.__qualname__

    # Handle built-ins (module is 'builtins')
    if module == "builtins":
        return qualname

    return f"{module}.{qualname}"


def cli_sanitise(value: str) -> str:
    return value.replace("_", "-")


def write_csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Serialise a list of row dicts to a CSV string.

    Only the columns listed in ``columns`` are written, in that order.
    Extra keys in each row are ignored; missing keys produce an empty cell.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
