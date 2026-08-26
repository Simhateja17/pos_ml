"""Idempotent forecast write-back; the UI consumes the existing reason contract.

Phase 8: a suggestion belongs to one SHOP. The database function resolves that
shop's stock and that shop's inbound orders, so store_id is required rather
than optional — a forecast written against the wrong shelf is worse than none.
"""
from __future__ import annotations
import json
import psycopg
from forecast import ForecastResult

def confidence(result: ForecastResult) -> str:
    width = (result.upper - result.lower) / max(result.demand, 1)
    return "high" if width <= .35 else "medium" if width <= .8 else "low"

def upsert(connection: psycopg.Connection, tenant_id: str, store_id: str, variant_id: str, result: ForecastResult, context: dict) -> None:
    with connection.cursor() as cur:
        cur.execute(
            """
            select public.ml_write_forecast_suggestion_v2(
              %s::uuid,%s::uuid,%s::uuid,%s::numeric,%s::numeric,%s::numeric,
              %s::numeric,%s::numeric,%s::numeric,%s::text,%s::numeric,%s::numeric,
              %s::integer,%s::integer,%s::integer,%s::integer,%s::integer,%s::numeric,%s::integer
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
                context['historyDays'],
                context['windowDays'],
                context['unitsSoldInWindow'],
                context['returnsInWindow'],
                context['netUnitsInWindow'],
                context['dailyVelocity'],
                context['reviewPeriodDays'],
            ),
        )
