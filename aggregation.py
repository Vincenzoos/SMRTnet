"""Pair-level aggregation matching SMRTnet's pinned inference threshold rule."""

from __future__ import print_function

import math


def aggregate_window_scores(scores):
    """Select one pair score from ordered 31-nt window probabilities.

    SMRTnet selects the maximum score at or above 0.5 when there are fewer than 40 windows.
    With 40 or more, a selected score must belong to a run of four consecutive windows at or
    above 0.5. Both cases fall back to the minimum score if no qualifying score exists.
    """

    values = [float(value) for value in scores]
    if not values or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("window scores must be non-empty finite probabilities in [0, 1]")
    if len(values) < 40:
        selected = [value for value in values if value >= 0.5]
        return max(selected) if selected else min(values)
    qualifying_positions = set()
    for start in range(len(values) - 3):
        if all(value >= 0.5 for value in values[start : start + 4]):
            qualifying_positions.update(range(start, start + 4))
    if not qualifying_positions:
        return min(values)
    return max(values[index] for index in qualifying_positions)


__all__ = ["aggregate_window_scores"]
