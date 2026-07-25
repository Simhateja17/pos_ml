"""Idempotent forecast write-back; the UI consumes the existing reason contract."""
from __future__ import annotations
import json
from math import ceil
import psycopg
from forecast import ForecastResult

def confidence(result: ForecastResult) -> str:
    width = (result.upper - result.lower) / max(result.demand, 1)
    return "high" if width <= .35 else "medium" if width <= .8 else "low"

def upsert(connection: psycopg.Connection, tenant_id: str, variant_id: str, result: ForecastResult, context: dict) -> None:
    safety = max(0, ceil(result.upper - result.demand))
    raw = result.demand + safety - context["currentStock"] - context["onOrder"]
    quantity = max(1, ceil(raw))
    reason = {**context, "formula": "forecast_interval_reorder", "forecastDemand": result.demand,
              "forecastLower": result.lower, "forecastUpper": result.upper, "model": result.model,
              "forecastWape": result.wape, "heuristicWape": result.heuristic_wape,
              "safetyStock": safety, "rawSuggestion": raw}
    with connection.cursor() as cur:
        cur.execute("""insert into public.reorder_suggestions
          (tenant_id,variant_id,supplier_id,suggested_quantity,reason,method,confidence,generated_at)
          values (%s,%s,%s,%s,%s::jsonb,'forecast',%s,now())
          on conflict (tenant_id,variant_id,((generated_at at time zone 'UTC')::date)) do update
          set suggested_quantity=excluded.suggested_quantity, reason=excluded.reason,
              method='forecast', confidence=excluded.confidence, generated_at=excluded.generated_at""",
          (tenant_id,variant_id,context.get("supplierId"),quantity,json.dumps(reason),confidence(result)))
