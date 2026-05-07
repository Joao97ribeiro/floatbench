"""Utilities for handling group values."""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence


def natural_key(text: str) -> List[Any]:
    """Helper function for natural sorting of section names.

    Args:
        text: Input string (e.g., section name).

    Returns:
        List of strings and integers for natural sorting.
    """

    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r'(\d+)', str(text))
    ]


def resolve_group_order(
    raw_groups: Iterable,
    group_order: Sequence | None = None,
) -> list:
    """Filter NaN/None from groups and apply optional ordering.

    Args:
        raw_groups: Iterable of raw group values (may contain
          None/NaN).
        group_order: Optional sequence defining preferred order.

    Returns:
        List of unique, ordered group values.
    """
    clean = [g for g in raw_groups if g is not None and str(g) != "nan"]
    if group_order is not None:
        order = list(group_order)
        unique = [g for g in order if g in clean]
        remaining = [g for g in clean if g not in unique]
        unique.extend(remaining)
        return unique
    return sorted(clean)
