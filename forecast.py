"""StatsForecast demand models and held-out evaluation for ML-02."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive


SEASON_LENGTH = 7
HOLDOUT_DAYS = 14


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
    model, wape = min(scores.items(), key=lambda item: item[1])
    if wape >= heuristic_wape:
        raise ValueError(f"forecast does not beat heuristic: WAPE {wape:.3f} >= {heuristic_wape:.3f}")

    future = _forecast(series, horizon)
    demand = max(0.0, float(future[model].sum()))
    lower_column, upper_column = f"{model}-lo-80", f"{model}-hi-80"
    lower = max(0.0, float(future[lower_column].sum()))
    upper = max(demand, float(future[upper_column].sum()))
    standard = max(0.0, float(future[model].head(14).sum()))
    return ForecastResult(
        model=model, demand=demand, lower=lower, upper=upper,
        wape=wape, heuristic_wape=heuristic_wape,
        standard_14_demand=standard, horizon_days=horizon,
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
    lower_column, upper_column = f"{base.model}-lo-80", f"{base.model}-hi-80"
    operational = future.head(horizon)
    demand = max(0.0, float(operational[base.model].sum()))
    lower = max(0.0, float(operational[lower_column].sum()))
    upper = max(demand, float(operational[upper_column].sum()))
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
    )
