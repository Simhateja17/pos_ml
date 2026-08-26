"""Claim and execute one queued manual/nightly forecast run.

The worker lives beside the Node backend on the GCP VM, but the two processes
only communicate through Postgres.  The ml_forecast role reads rollups and
uses narrowly scoped SECURITY DEFINER adapters for queue/context/writeback
operations.  It never receives the backend runtime credential.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from typing import Any

import pandas as pd
import psycopg
import statsforecast
from psycopg.rows import dict_row

from eligibility import assess
from forecast import calendar_metrics, daily_series, evaluate_and_forecast_profile
from job import _connect, _psycopg_url


REVIEW_PERIOD_DAYS = 7
WINDOW_DAYS = 30
WORKER_VERSION = "manual-worker-1"


@dataclass(frozen=True)
class WorkerSettings:
    """Connection settings for the multi-tenant queue worker.

    The nightly job still receives one ML_TENANT_ID. The manual worker does
    not: its queue claim returns the tenant to process, after which every
    tenant-scoped adapter sets and validates app.tenant_id.
    """

    database_url: str

    @classmethod
    def from_environment(cls) -> "WorkerSettings":
        database_url = os.environ.get("ML_DATABASE_URL", "").strip()
        if not database_url:
            raise ValueError("ML_DATABASE_URL is required")
        return cls(database_url=_psycopg_url(database_url))


def _json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _set_tenant(connection: psycopg.Connection, tenant_id: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute("select set_config('app.tenant_id', %s, true)", (tenant_id,))


def _claim(connection: psycopg.Connection) -> dict[str, Any] | None:
    with connection.transaction():
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select * from public.ml_claim_next_forecast_run()")
            return cursor.fetchone()


def _context(connection: psycopg.Connection, tenant_id: str, run_id: str) -> list[dict[str, Any]]:
    with connection.transaction():
        _set_tenant(connection, tenant_id)
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "select * from public.ml_forecast_run_context(%s::uuid,%s::uuid)",
                (tenant_id, run_id),
            )
            return list(cursor.fetchall())


def _rollup(connection: psycopg.Connection, tenant_id: str, store_id: str) -> pd.DataFrame:
    with connection.transaction():
        _set_tenant(connection, tenant_id)
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                select variant_id, date, units_sold, returns_units
                from public.daily_sales_rollup
                where tenant_id = %s::uuid and store_id = %s::uuid
                order by variant_id, date
                """,
                (tenant_id, store_id),
            )
            return pd.DataFrame(cursor.fetchall())


def _stockout_dates(
    connection: psycopg.Connection,
    tenant_id: str,
    store_id: str,
    variant_id: str,
    start_date: Any,
    end_date: Any,
) -> set[pd.Timestamp]:
    with connection.transaction():
        _set_tenant(connection, tenant_id)
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "select stockout_date from public.ml_stockout_dates(%s::uuid,%s::uuid,%s::uuid,%s::date,%s::date)",
                (tenant_id, store_id, variant_id, start_date, end_date),
            )
            return {pd.Timestamp(row["stockout_date"]) for row in cursor.fetchall()}


def _heuristic(
    group: pd.DataFrame,
    context: dict[str, Any],
    as_of_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    dates = pd.to_datetime(group["date"])
    observed_start = dates.min()
    observed_end = dates.max()
    end_date = pd.Timestamp(as_of_date) if as_of_date is not None else observed_end
    if end_date.tzinfo is not None:
        end_date = end_date.tz_localize(None)
    recent = group.loc[dates >= end_date - pd.Timedelta(days=WINDOW_DAYS - 1)]
    net = (group["units_sold"] - group["returns_units"]).clip(lower=0)
    recent_net = (recent["units_sold"] - recent["returns_units"]).clip(lower=0)
    history_days = int((observed_end - observed_start).days) + 1
    units = float(recent["units_sold"].sum())
    returns = float(recent["returns_units"].sum())
    net_units = float(recent_net.sum())
    effective_days = max(1, min(history_days, WINDOW_DAYS))
    velocity = net_units / effective_days
    lead_days = max(0, int(context.get("lead_time_days") or 7))
    review_days = max(0, int(context.get("review_period_days") or REVIEW_PERIOD_DAYS))
    lead_demand = velocity * lead_days
    safety_days = 7
    safety = velocity * safety_days
    review_demand = velocity * review_days
    stock = float(context.get("current_stock") or 0)
    on_order = float(context.get("on_order") or 0)
    reorder_point = lead_demand + safety
    raw = reorder_point + review_demand - stock - on_order
    return {
        "formula": "velocity_x_lead_time",
        "windowDays": WINDOW_DAYS,
        "historyDays": history_days,
        "unitsSoldInWindow": int(units),
        "returnsInWindow": int(returns),
        "netUnitsInWindow": int(net_units),
        "dailyVelocity": velocity,
        "leadTimeDays": lead_days,
        "leadTimeDemand": lead_demand,
        "safetyDays": safety_days,
        "safetyStock": safety,
        "reorderPoint": reorder_point,
        "reviewPeriodDays": review_days,
        "reviewPeriodDemand": review_demand,
        "currentStock": int(stock),
        "onOrder": int(on_order),
        "rawSuggestion": raw,
        "suggestedQuantity": max(0, int(ceil(raw))),
        "supplierName": context.get("supplier_name"),
        "basis": "this_store",
        "totalNetUnits": float(net.sum()),
    }


def _write_item(
    connection: psycopg.Connection,
    tenant_id: str,
    run_id: str,
    store_id: str,
    variant_id: str,
    history_days: int | None,
    trailing_units: float,
    total_units: float,
    eligible: bool,
    context: dict[str, Any],
    rule_based: dict[str, Any],
    ml_result: dict[str, Any],
    disposition: str,
    reason_code: str | None,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select public.ml_write_forecast_run_item(
              %s::uuid,%s::uuid,%s::uuid,%s::uuid,%s::integer,
              %s::numeric,%s::numeric,%s::boolean,%s::uuid,%s::integer,
              %s::integer,%s::integer,%s::jsonb,%s::jsonb,
              %s::public.forecast_run_item_disposition,%s::text
            )
            """,
            (
                tenant_id,
                run_id,
                store_id,
                variant_id,
                history_days,
                trailing_units,
                total_units,
                eligible,
                context.get("supplier_id"),
                context.get("lead_time_days"),
                context.get("review_period_days") or REVIEW_PERIOD_DAYS,
                (context.get("lead_time_days") or 7) + (context.get("review_period_days") or REVIEW_PERIOD_DAYS),
                _json(rule_based),
                _json(ml_result),
                disposition,
                reason_code,
            ),
        )


def _write_forecast_suggestion(
    connection: psycopg.Connection,
    tenant_id: str,
    store_id: str,
    variant_id: str,
    result: Any,
    heuristic: dict[str, Any],
    context: dict[str, Any],
    generated_at: datetime,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select public.ml_write_forecast_suggestion_v3(
              %s::uuid,%s::uuid,%s::uuid,%s::numeric,%s::numeric,%s::numeric,
              %s::numeric,%s::numeric,%s::numeric,%s::text,%s::numeric,%s::numeric,
              %s::integer,%s::integer,%s::integer,%s::integer,%s::integer,
              %s::numeric,%s::integer,%s::timestamptz
            )
            """,
            (
                tenant_id,
                store_id,
                variant_id,
                result.demand,
                result.lower,
                result.upper,
                result.lead_time_demand,
                result.review_period_demand,
                result.standard_14_demand,
                result.model,
                result.wape,
                result.heuristic_wape,
                int(heuristic["historyDays"]),
                WINDOW_DAYS,
                int(heuristic["unitsSoldInWindow"]),
                int(heuristic["returnsInWindow"]),
                int(heuristic["netUnitsInWindow"]),
                heuristic["dailyVelocity"],
                int(context.get("review_period_days") or REVIEW_PERIOD_DAYS),
                generated_at,
            ),
        )


def _complete(
    connection: psycopg.Connection,
    tenant_id: str,
    run_id: str,
    counts: dict[str, int],
) -> None:
    with connection.transaction():
        _set_tenant(connection, tenant_id)
        with connection.cursor() as cursor:
            cursor.execute(
                "select public.ml_complete_forecast_run(%s::uuid,%s::uuid,%s::integer,%s::integer,%s::integer,%s::integer,%s::integer,%s::text,%s::text)",
                (
                    tenant_id,
                    run_id,
                    counts["evaluated"],
                    counts["eligible"],
                    counts["won"],
                    counts["written"],
                    counts["skipped"],
                    WORKER_VERSION,
                    statsforecast.__version__,
                ),
            )


def _fail(connection: psycopg.Connection, tenant_id: str, run_id: str, code: str, message: str) -> None:
    with connection.transaction():
        _set_tenant(connection, tenant_id)
        with connection.cursor() as cursor:
            cursor.execute(
                "select public.ml_fail_forecast_run(%s::uuid,%s::uuid,%s::text,%s::text,%s::text)",
                (tenant_id, run_id, code[:80], message[:1000], WORKER_VERSION),
            )


def _heartbeat(connection: psycopg.Connection, tenant_id: str, run_id: str) -> None:
    with connection.transaction():
        _set_tenant(connection, tenant_id)
        with connection.cursor() as cursor:
            cursor.execute(
                "select public.ml_touch_forecast_run(%s::uuid,%s::uuid)",
                (tenant_id, run_id),
            )


def _eligibility_metrics(group: pd.DataFrame, as_of_date: Any) -> tuple[int, float, float]:
    return calendar_metrics(group, pd.Timestamp(as_of_date))


def execute_run(connection: psycopg.Connection, tenant_id: str, run: dict[str, Any]) -> dict[str, int]:
    run_id = str(run["run_id"])
    store_id = str(run["store_id"])
    as_of_date = pd.Timestamp(run["requested_at"]).normalize()
    rows = _rollup(connection, tenant_id, store_id)
    contexts = _context(connection, tenant_id, run_id)
    context_by_variant = {str(row["variant_id"]): row for row in contexts}
    counts = {"evaluated": 0, "eligible": 0, "won": 0, "written": 0, "skipped": 0}
    # Every forecast suggestion from one run must carry the same generation
    # timestamp. Each item is deliberately committed in its own short
    # transaction, so the database function cannot rely on transaction-local
    # now() to identify the logical batch.
    generated_at = pd.Timestamp.now(tz="UTC").to_pydatetime()

    for variant_id, group in rows.groupby("variant_id") if not rows.empty else []:
        variant_id = str(variant_id)
        counts["evaluated"] += 1
        if counts["evaluated"] % 10 == 0:
            _heartbeat(connection, tenant_id, run_id)
        context = context_by_variant.get(variant_id, {})
        history_days, trailing_units, total_units = _eligibility_metrics(group, as_of_date)
        heuristic = _heuristic(group, context, as_of_date)
        if not context.get("supplier_id"):
            counts["skipped"] += 1
            with connection.transaction():
                _set_tenant(connection, tenant_id)
                _write_item(connection, tenant_id, run_id, store_id, variant_id, history_days, trailing_units, total_units, False, context, heuristic, {}, "no_supplier", "no_supplier")
            continue

        gate = assess(history_days, int(trailing_units), int(total_units))
        if not gate.eligible:
            counts["skipped"] += 1
            with connection.transaction():
                _set_tenant(connection, tenant_id)
                _write_item(connection, tenant_id, run_id, store_id, variant_id, history_days, trailing_units, total_units, False, context, heuristic, {"eligibility": gate.reason}, "ineligible", gate.reason)
            continue

        counts["eligible"] += 1
        stockouts = _stockout_dates(connection, tenant_id, store_id, variant_id, group["date"].min(), group["date"].max())
        try:
            result = evaluate_and_forecast_profile(
                daily_series(group, stockouts, end_date=as_of_date),
                int(context.get("lead_time_days") or 7),
                int(context.get("review_period_days") or REVIEW_PERIOD_DAYS),
            )
        except ValueError as error:
            reason = "model_not_better" if "does not beat heuristic" in str(error) else "forecast_error"
            counts["skipped"] += 1
            with connection.transaction():
                _set_tenant(connection, tenant_id)
                _write_item(connection, tenant_id, run_id, store_id, variant_id, history_days, trailing_units, total_units, True, context, heuristic, {"error": reason}, "heuristic_won" if reason == "model_not_better" else "failed", reason)
            continue

        safety = max(0, ceil(result.upper - result.demand))
        ml_raw = result.demand + safety - float(context.get("current_stock") or 0) - float(context.get("on_order") or 0)
        ml_quantity = max(0, int(ceil(ml_raw)))
        ml_result = {
            "model": result.model,
            "forecastDemand": result.demand,
            "forecast14DayDemand": result.standard_14_demand,
            "forecastLower": result.lower,
            "forecastUpper": result.upper,
            "forecastHorizonDays": result.horizon_days,
            "forecastWape": result.wape,
            "heuristicWape": result.heuristic_wape,
            "confidence": "high" if (result.upper - result.lower) / max(result.demand, 1) <= .35 else "medium" if (result.upper - result.lower) / max(result.demand, 1) <= .8 else "low",
            "safetyStock": safety,
            "rawSuggestion": ml_raw,
            "suggestedQuantity": ml_quantity,
        }
        disposition = "forecast_written" if ml_quantity > 0 else "sufficient_stock"
        with connection.transaction():
            _set_tenant(connection, tenant_id)
            if ml_quantity > 0:
                _write_forecast_suggestion(connection, tenant_id, store_id, variant_id, result, heuristic, context, generated_at)
                counts["written"] += 1
            counts["won"] += 1
            _write_item(connection, tenant_id, run_id, store_id, variant_id, history_days, trailing_units, total_units, True, context, heuristic, ml_result, disposition, None)

    return counts


def main() -> int:
    try:
        settings = WorkerSettings.from_environment()
        with _connect(settings.database_url) as connection:
            run = _claim(connection)
            if not run:
                print(json.dumps({"status": "idle", "scope": "all_tenants"}))
                return 0
            run_id = str(run["run_id"])
            tenant_id = str(run["tenant_id"])
            try:
                counts = execute_run(connection, tenant_id, run)
                _complete(connection, tenant_id, run_id, counts)
                print(json.dumps({"status": "completed", "run_id": run_id, **counts}, sort_keys=True))
                return 0
            except Exception as error:
                # Keep database-facing evidence safe for the owner-facing API;
                # the raw exception is still available in journalctl.
                print(f"manual forecast run {run_id} failed: {error}", file=sys.stderr)
                _fail(
                    connection,
                    tenant_id,
                    run_id,
                    "worker_error",
                    "The forecast worker could not complete this run. Check the VM journal for details.",
                )
                raise
    except (ValueError, PermissionError, psycopg.Error) as error:
        print(f"manual forecast worker failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"manual forecast worker failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
