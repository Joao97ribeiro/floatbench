"""Preset seed and wind/wave index helpers."""

from typing import List, Literal


def get_seed_subset(n: int) -> List[str]:
    """Return seed labels S1..S{n}, clipped to 1..6.

    Args:
        n: Number of seeds to keep.

    Returns:
        List of labels like ["S1", "S2", ...].
    """
    n = max(0, min(int(n), 6))
    return [f"S{i}" for i in range(1, n + 1)]


def get_wind_wave_indices(
    selection: int,
    mode: Literal["wind", "waves"] = "wind",
) -> List[int]:
    """Pick preset indices for wind or wave conditions.

    Args:
        selection: Preset key (e.g., 11 for wind; 3 for waves).
        mode: Either ``"wind"`` or ``"waves"``.

    Returns:
        List of valid indices.
    """
    presets = {
        "wind": {
            1: [10],
            2: [5, 15],
            3: [0, 10, 21],
            4: [0, 7, 14, 21],
            5: [0, 5, 10, 15, 20],
            6: [0, 4, 8, 12, 16, 20],
            8: [0, 3, 5, 7, 9, 12, 15, 18, 21],
            11: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            22:
                list(range(22)),
            -1: [
                1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20
            ],
        },
        "waves": {
            1: [3],
            2: [1, 5],
            3: [0, 3, 6],
            4: [1, 2, 4, 5],  # FLOATBench release default (paper Table F.1)
            5: [0, 1, 3, 5, 6],
            6: [0, 1, 2, 4, 5, 6],
            7: list(range(7)),
            -1: list(range(7)),
        },
    }
    if mode not in presets:
        raise ValueError(f"mode must be 'wind' or 'waves', got {mode!r}")
    if selection not in presets[mode]:
        raise ValueError(
            f"selection={selection} not a preset for mode={mode!r}; "
            f"valid: {sorted(presets[mode].keys())}")
    return presets[mode][selection]
