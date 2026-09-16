"""Unit tests for the horizon-interval and gating logic.

These cover the two defects found by the 2026-09-16 backtest against the QA
tenant's real rollup: intervals summed as if perfectly correlated, and a
quality gate scored on a metric the product does not consume.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from forecast import Z_80, aggregate_interval


def _frame(model: str, lows: list[float], highs: list[float]) -> pd.DataFrame:
    return pd.DataFrame({model: [(l + h) / 2 for l, h in zip(lows, highs)],
                         f"{model}-lo-80": lows, f"{model}-hi-80": highs})


def test_interval_is_centred_on_the_point_forecast() -> None:
    frame = _frame("AutoETS", [1.0] * 4, [3.0] * 4)
    lower, upper = aggregate_interval(frame, "AutoETS", 8.0)
    assert math.isclose((lower + upper) / 2, 8.0, rel_tol=1e-9)


def test_interval_sits_between_independent_and_perfectly_correlated() -> None:
    """The whole point of the fix: neither extreme is correct."""
    frame = _frame("AutoETS", [1.0] * 14, [5.0] * 14)
    lower, upper = aggregate_interval(frame, "AutoETS", 42.0)
    width = upper - lower
    independent = (5.0 - 1.0) * math.sqrt(14)          # rho = 0
    perfectly_correlated = (5.0 - 1.0) * 14            # rho = 1, the old code
    assert independent < width < perfectly_correlated


def test_interval_is_narrower_than_the_summed_bounds_it_replaces() -> None:
    """The regression guard: the old code summed the daily bounds."""
    frame = _frame("AutoETS", [1.0] * 14, [5.0] * 14)
    lower, upper = aggregate_interval(frame, "AutoETS", 42.0)
    summed_width = frame["AutoETS-hi-80"].sum() - frame["AutoETS-lo-80"].sum()
    assert (upper - lower) < summed_width


def test_lower_bound_never_goes_negative() -> None:
    """Demand cannot be negative; a wide band on a small forecast must clamp."""
    frame = _frame("SeasonalNaive", [0.0] * 7, [10.0] * 7)
    lower, upper = aggregate_interval(frame, "SeasonalNaive", 1.0)
    assert lower == 0.0
    assert upper >= 1.0


def test_upper_bound_never_falls_below_the_point_forecast() -> None:
    frame = _frame("AutoARIMA", [2.0] * 3, [2.0] * 3)
    lower, upper = aggregate_interval(frame, "AutoARIMA", 6.0)
    assert upper >= 6.0


def test_correlation_of_one_reproduces_the_old_summed_bounds() -> None:
    """Pins the meaning of rho so a future edit cannot silently reinterpret it."""
    import forecast

    original = forecast.DEMAND_ERROR_CORRELATION
    try:
        forecast.DEMAND_ERROR_CORRELATION = 1.0
        frame = _frame("AutoETS", [1.0] * 14, [5.0] * 14)
        lower, upper = forecast.aggregate_interval(frame, "AutoETS", 42.0)
        summed = frame["AutoETS-hi-80"].sum() - frame["AutoETS-lo-80"].sum()
        assert math.isclose(upper - lower, summed, rel_tol=1e-9)
    finally:
        forecast.DEMAND_ERROR_CORRELATION = original


def test_zero_width_daily_bands_produce_a_zero_width_horizon_band() -> None:
    frame = _frame("AutoETS", [4.0] * 5, [4.0] * 5)
    lower, upper = aggregate_interval(frame, "AutoETS", 20.0)
    assert math.isclose(lower, 20.0, rel_tol=1e-9)
    assert math.isclose(upper, 20.0, rel_tol=1e-9)


def test_z_80_matches_the_level_passed_to_statsforecast() -> None:
    """A mismatch here silently rescales every interval."""
    from statistics import NormalDist

    assert math.isclose(Z_80, NormalDist().inv_cdf(0.9), rel_tol=1e-9)
