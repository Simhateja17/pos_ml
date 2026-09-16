"""StatsForecast demand models and held-out evaluation for ML-02."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive


SEASON_LENGTH = 7
HOLDOUT_DAYS = 14

# Two-sided 80% quantile of the standard normal, matching level=[80].
Z_80 = 1.2815515655446004

# Average pairwise correlation between one variant's daily forecast errors.
#
# PROVISIONAL. The two defensible closed forms are both wrong in practice:
# treating days as independent (rho = 0) gave 49% empirical coverage for a
# nominal 80% band, and summing the daily bounds (rho = 1, the original code)
# gave 99%. Real errors are partially correlated -- a week that runs hot tends
# to keep running hot -- so the truth sits between them.
#
# 0.15 is a round number near the value that reproduced ~80% coverage on the
# 2026-09-16 QA backtest. That backtest ran on ultra-sparse data (median 1 unit
# per variant per fortnight), so this is an order-of-magnitude estimate, NOT a
# calibrated constant. Re-fit it against the first pilot shop's real history
# before trusting the safety stock derived from it.
DEMAND_ERROR_CORRELATION = 0.15


@dataclass(frozen=True)
class ForecastResult:
    model: str
    demand: float
    lower: float
    upper: float
    wape: float
    heuristic_wape: float
    standard_14_demand: float = 0.0
    lead_time_demand: float = 0.0
    review_period_demand: float = 0.0
    horizon_days: int = 14
    total_error: float = 0.0
    heuristic_total_error: float = 0.0


def daily_series(
    rows: pd.DataFrame,
    stockout_dates: set[pd.Timestamp],
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Create one complete daily series and impute only ledger-proven stockouts."""
    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    units_sold = pd.to_numeric(frame["units_sold"], errors="coerce").fillna(0.0)
    returns_units = pd.to_numeric(frame["returns_units"], errors="coerce").fillna(0.0)
    frame["net_units"] = (units_sold - returns_units).clip(lower=0).astype(float)
    observed_end = frame["date"].max()
    requested_end = pd.Timestamp(end_date) if end_date is not None else observed_end
    if requested_end.tzinfo is not None:
        requested_end = requested_end.tz_localize(None)
    all_dates = pd.date_range(frame["date"].min(), max(observed_end, requested_end), freq="D")
    result = frame.set_index("date").reindex(all_dates).rename_axis("ds").reset_index()
    result["unique_id"] = str(frame["variant_id"].iloc[0])
    result["y"] = result["net_units"].fillna(0.0)

    # A true stockout zero is censored demand. Estimate it from the same weekday
    # in observed weeks; use the observed mean only when that weekday is absent.
    observed = result.loc[~result["ds"].isin(stockout_dates), ["ds", "y"]].copy()
    observed["weekday"] = observed["ds"].dt.dayofweek
    weekday_mean = observed.groupby("weekday")["y"].mean()
    fallback = float(observed["y"].mean()) if not observed.empty else 0.0
    mask = result["ds"].isin(stockout_dates)
    result.loc[mask, "y"] = result.loc[mask, "ds"].dt.dayofweek.map(weekday_mean).fillna(fallback)
    return result[["unique_id", "ds", "y"]]


def calendar_metrics(
    rows: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> tuple[int, float, float]:
    """Return history span, recent net units, and total net units.

    ``daily_sales_rollup`` is sparse: it stores activity days only. The
    eligibility thresholds are calendar-based, so source-row counts would
    reject mature intermittent products and an active-row tail would not be a
    true recent-calendar window.
    """

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    observed_start = frame["date"].min()
    observed_end = frame["date"].max()
    series = daily_series(frame, set(), end_date=as_of_date)
    history_days = int((observed_end - observed_start).days) + 1
    trailing_units = float(series.tail(14)["y"].sum())
    total_units = float(series["y"].sum())
    return history_days, trailing_units, total_units


def _forecast(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    models = [
        AutoETS(season_length=SEASON_LENGTH, alias="AutoETS"),
        AutoARIMA(season_length=SEASON_LENGTH, alias="AutoARIMA"),
        SeasonalNaive(season_length=SEASON_LENGTH, alias="SeasonalNaive"),
    ]
    return StatsForecast(models=models, freq="D", n_jobs=1).forecast(horizon, train, level=[80])


def aggregate_interval(frame: pd.DataFrame, model: str, mean_total: float) -> tuple[float, float]:
    """Combine per-day 80% intervals into one interval for the horizon total.

    ``statsforecast`` reports an interval per DAY. Adding the daily bounds
    together assumes every day misses in the same direction at once — the
    perfectly-correlated worst case. Measured against the QA tenant's real
    rollup, that produced 99% empirical coverage for a nominal 80% band, i.e.
    an interval roughly five times the point forecast.

    That width is not free: ``job.py`` derives safety stock from it, so an
    over-wide band silently inflates every reorder point and tells owners to
    over-order — the exact failure this feature exists to remove.

    Independent daily errors would combine in quadrature, growing with sqrt(n)
    rather than n -- but measured coverage showed that over-corrects (49% for a
    nominal 80%). Days are partially correlated, so this applies the standard
    equicorrelated-sum variance:

        Var(sum) = sum(sigma_i^2) * (1 + (n - 1) * rho)

    rho = 0 recovers quadrature, rho = 1 recovers the original summed bounds.
    See DEMAND_ERROR_CORRELATION for why the chosen value is provisional.
    """

    lower = frame[f"{model}-lo-80"].to_numpy(float)
    upper = frame[f"{model}-hi-80"].to_numpy(float)
    daily_sigma = (upper - lower) / (2.0 * Z_80)
    days = len(daily_sigma)
    inflation = 1.0 + (days - 1) * DEMAND_ERROR_CORRELATION
    horizon_sigma = float(np.sqrt(np.square(daily_sigma).sum() * inflation))
    half_width = Z_80 * horizon_sigma
    return max(0.0, mean_total - half_width), max(mean_total, mean_total + half_width)


def evaluate_and_forecast(series: pd.DataFrame, horizon: int) -> ForecastResult:
    """Select a model only when it beats the Phase 5 velocity baseline on WAPE."""
    if len(series) <= HOLDOUT_DAYS:
        raise ValueError("Series is too short for a held-out evaluation")
    train = series.iloc[:-HOLDOUT_DAYS]
    actual = series.iloc[-HOLDOUT_DAYS:]["y"].to_numpy(float)
    held = _forecast(train, HOLDOUT_DAYS)
    denominator = max(float(actual.sum()), 1.0)
    heuristic = float(train["y"].mean())
    heuristic_wape = float(abs(actual - heuristic).sum() / denominator)
    candidate_columns = ["AutoETS", "AutoARIMA", "SeasonalNaive"]
    scores = {name: float(abs(actual - held[name].to_numpy(float)).sum() / denominator) for name in candidate_columns}
    # Selection stays on per-day WAPE: it reads every day of the holdout, so a
    # model cannot win it by having two large errors cancel out.
    model, wape = min(scores.items(), key=lambda item: item[1])
    if wape >= heuristic_wape:
        raise ValueError(f"forecast does not beat heuristic: WAPE {wape:.3f} >= {heuristic_wape:.3f}")

    # The gate, however, must also check the number the product actually uses.
    # Every reorder decision consumes the HORIZON TOTAL, not the daily path,
    # and the two disagree: on the QA backtest, models cleared the per-day gate
    # and then lost to the velocity heuristic on the total that drives the
    # purchase order. Winning per-day but losing on the total is not a win.
    actual_total = float(actual.sum())
    total_error = abs(float(held[model].to_numpy(float).sum()) - actual_total) / denominator
    heuristic_total_error = abs(heuristic * HOLDOUT_DAYS - actual_total) / denominator
    if total_error >= heuristic_total_error:
        raise ValueError(
            f"forecast does not beat heuristic on horizon total: "
            f"{total_error:.3f} >= {heuristic_total_error:.3f}"
        )

    future = _forecast(series, horizon)
    demand = max(0.0, float(future[model].sum()))
    lower, upper = aggregate_interval(future, model, demand)
    standard = max(0.0, float(future[model].head(14).sum()))
    return ForecastResult(
        model=model, demand=demand, lower=lower, upper=upper,
        wape=wape, heuristic_wape=heuristic_wape,
        standard_14_demand=standard, horizon_days=horizon,
        total_error=total_error, heuristic_total_error=heuristic_total_error,
    )


def evaluate_and_forecast_profile(
    series: pd.DataFrame,
    lead_time_days: int,
    review_period_days: int = 7,
) -> ForecastResult:
    """Evaluate the model once and retain operational and standard horizons.

    The model is selected using the same held-out WAPE gate as the nightly
    job. The future frame is long enough to split the operational horizon into
    supplier lead-time demand and the review buffer, while the first 14 days
    remain available as a stable comparison metric for the test UI.
    """

    horizon = max(1, int(lead_time_days) + int(review_period_days))
    # Reuse the existing evaluator for the quality gate and model choice. It
    # also produces the exact operational aggregate, avoiding two subtly
    # different selection paths.
    base = evaluate_and_forecast(series, horizon)
    future = _forecast(series, max(14, horizon))
    operational = future.head(horizon)
    demand = max(0.0, float(operational[base.model].sum()))
    lower, upper = aggregate_interval(operational, base.model, demand)
    lead = future.head(max(0, int(lead_time_days)))
    review = future.iloc[max(0, int(lead_time_days)):horizon]
    return ForecastResult(
        model=base.model,
        demand=demand,
        lower=lower,
        upper=upper,
        wape=base.wape,
        heuristic_wape=base.heuristic_wape,
        standard_14_demand=max(0.0, float(future.head(14)[base.model].sum())),
        lead_time_demand=max(0.0, float(lead[base.model].sum())),
        review_period_demand=max(0.0, float(review[base.model].sum())),
        horizon_days=horizon,
        total_error=base.total_error,
        heuristic_total_error=base.heuristic_total_error,
    )
