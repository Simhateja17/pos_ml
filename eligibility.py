"""Explainable, per-variant maturity gate for ML-02."""

from __future__ import annotations

from dataclasses import dataclass


MIN_HISTORY_DAYS = 60
TRAILING_DAYS = 14
MIN_TRAILING_UNITS = 1
MIN_TOTAL_UNITS = 30


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    history_days: int
    trailing_units: int
    total_units: int
    reason: str


def assess(history_days: int, trailing_units: int, total_units: int) -> Eligibility:
    if history_days < MIN_HISTORY_DAYS:
        return Eligibility(False, history_days, trailing_units, total_units, "insufficient_history")
    if trailing_units < MIN_TRAILING_UNITS:
        return Eligibility(False, history_days, trailing_units, total_units, "no_recent_sales")
    if total_units < MIN_TOTAL_UNITS:
        return Eligibility(False, history_days, trailing_units, total_units, "insufficient_volume")
    return Eligibility(True, history_days, trailing_units, total_units, "eligible")
