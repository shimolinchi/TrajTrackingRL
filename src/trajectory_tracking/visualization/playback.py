"""Simulation-time clip selection, independent of recording and rendering."""
from math import isfinite


def advance_clip(index, seconds, durations, hold_seconds=0.5):
    """Return clip index/time, preserving elapsed time across every loop."""
    if (not durations or not 0 <= index < len(durations)
            or not isfinite(seconds) or seconds < 0
            or not isfinite(hold_seconds) or hold_seconds < 0
            or any(not isfinite(d) or d <= 0 for d in durations)):
        raise ValueError("invalid clip timing")
    seconds %= sum(durations) + len(durations) * hold_seconds
    while seconds >= durations[index] + hold_seconds:
        seconds -= durations[index] + hold_seconds
        index = (index + 1) % len(durations)
    return index, seconds
