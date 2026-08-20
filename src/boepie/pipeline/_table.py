"""Shared CSV rendering for output family F1.

F1 (`design/interface-spec.md`) is the table of homogeneous records: a count
line followed by CSV with a header row, emitted by `list_cabs`,
`get_cab_params` and `get_ms_fields`. CSV rather than markdown because the
header amortises the field names across N rows instead of repeating them per
row - which matters, since every line here is paid on every call of every
session.
"""

from __future__ import annotations

import csv
import io
from typing import Any


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
