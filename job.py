"""Nightly forecasting job entry point.

Task 1 establishes and verifies the database seam only. Forecasting and
write-back are added by the later Phase 6 tasks; Express never imports or calls
this module.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID

import pandas as pd
import psycopg
import statsforecast
from psycopg.rows import dict_row
from eligibility import assess
from forecast import daily_series, evaluate_and_forecast_profile
from writeback import upsert


ALLOWED_TABLE_PRIVILEGES: Final[dict[str, frozenset[str]]] = {
    "daily_sales_rollup": frozenset({"SELECT"}),
    "reorder_suggestions": frozenset({"INSERT", "UPDATE"}),
}


@dataclass(frozen=True)
class Settings:
    database_url: str
    tenant_id: UUID

    @classmethod
    def from_environment(cls) -> "Settings":
        database_url = os.environ.get("ML_DATABASE_URL", "").strip()
        tenant_id = os.environ.get("ML_TENANT_ID", "").strip()
        if not database_url:
            raise ValueError("ML_DATABASE_URL is required")
        if not tenant_id:
            raise ValueError("ML_TENANT_ID is required")
        return cls(database_url=_psycopg_url(database_url), tenant_id=UUID(tenant_id))


def _psycopg_url(database_url: str) -> str:
    """Drop Prisma-only pooler flags while preserving standard libpq options."""

    parsed = urlsplit(database_url)
    query = urlencode([
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "pgbouncer"
    ])
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment))


def _connect(database_url: str) -> psycopg.Connection:
    """Open a connection compatible with Supavisor transaction pooling.

    The ML role uses the transaction-mode pooler on port 6543.  Automatic
    server-side prepared statements are connection/backend-specific there and
    can collide when a transaction is assigned to another backend.
    """

    return psycopg.connect(database_url, prepare_threshold=None)


def _assert_least_privilege(connection: psycopg.Connection) -> None:
    """Fail closed if the ML role can touch any unapproved public table."""

    privilege_checks = ", ".join(
        f"has_table_privilege(current_user, format('public.%I', table_name), '{privilege}')"
        f" as {privilege.lower()}"
        for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")
    )
    query = f"""
        select table_name, {privilege_checks}
        from information_schema.tables
        where table_schema = 'public' and table_type = 'BASE TABLE'
        order by table_name
    """

    violations: list[str] = []
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(query)
        for row in cursor.fetchall():
            actual = {
                privilege.upper()
                for privilege in ("select", "insert", "update", "delete", "truncate", "references", "trigger")
                if row[privilege]
            }
            expected = ALLOWED_TABLE_PRIVILEGES.get(row["table_name"], frozenset())
            if actual != expected:
                violations.append(
                    f"{row['table_name']}: expected {sorted(expected)}, got {sorted(actual)}"
                )

    if violations:
        raise PermissionError("ML database role violates its allow-list: " + "; ".join(violations))


def inspect_rollup(settings: Settings) -> dict[str, object]:
    """Read one tenant's rollup and prove the connection role is constrained."""

    with _connect(settings.database_url) as connection:
        with connection.transaction():
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute("select set_config('app.tenant_id', %s, true)", (str(settings.tenant_id),))
                cursor.execute(
                    """
                    select
                      count(*)::integer as row_count,
                      count(distinct variant_id)::integer as variant_count,
                      min(date) as first_date,
                      max(date) as last_date
                    from public.daily_sales_rollup
                    """
                )
                summary = dict(cursor.fetchone())

            _assert_least_privilege(connection)
            summary["database_role"] = connection.info.user.split(".", 1)[0]
            summary["tenant_id"] = str(settings.tenant_id)
            summary["pandas_version"] = pd.__version__
            summary["statsforecast_version"] = statsforecast.__version__
            return summary


def main() -> int:
    try:
        settings = Settings.from_environment()
        summary = inspect_rollup(settings)
        with _connect(settings.database_url) as connection, connection.transaction():
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute("select set_config('app.tenant_id', %s, true)", (str(settings.tenant_id),))
                # Phase 8: the rollup is keyed (tenant, STORE, variant, date). Grouping
                # by variant alone yields one row per store per date, and
                # daily_series() reindexes on date — duplicate dates raise. The
                # job would crash outright for any business with two shops.
                #
                # Grouping per (store, variant) is also the correct MODEL: a shop
                # forecasts its own shelf. Andheri's demand curve is not Bandra's,
                # and averaging them produces a number that fits neither.
                cursor.execute("select store_id,variant_id,date,units_sold,returns_units from public.daily_sales_rollup order by store_id,variant_id,date")
                rows = pd.DataFrame(cursor.fetchall())
                written = 0
                if rows.empty:
                    summary['forecast_rows_written'] = 0
                    print(json.dumps(summary, default=str, sort_keys=True))
                    return 0
                for (store_id, variant_id), group in rows.groupby(['store_id', 'variant_id']):
                    history = len(group)
                    recent = group.tail(30)
                    trailing = int((group.tail(14).units_sold - group.tail(14).returns_units).clip(lower=0).sum())
                    total = int((group.units_sold - group.returns_units).clip(lower=0).sum())
                    if not assess(history, trailing, total).eligible:
                        continue
                    cursor.execute(
                        'select * from public.ml_forecast_variant_context(%s::uuid,%s::uuid,%s::uuid)',
                        (str(settings.tenant_id), str(store_id), str(variant_id)),
                    )
                    db_context = cursor.fetchone()
                    if not db_context or db_context['supplier_id'] is None:
                        continue
                    cursor.execute(
                        'select stockout_date from public.ml_stockout_dates(%s,%s,%s,%s,%s)',
                        (str(settings.tenant_id), str(store_id), str(variant_id), group.date.min(), group.date.max()),
                    )
                    stockouts = {pd.Timestamp(r['stockout_date']) for r in cursor.fetchall()}
                    try:
                        result = evaluate_and_forecast_profile(
                            daily_series(group, stockouts),
                            int(db_context['lead_time_days'] or 7),
                            7,
                        )
                    except ValueError:
                        continue
                    recent_net = (recent.units_sold - recent.returns_units).clip(lower=0)
                    effective_days = max(1, min(history, 30))
                    context = {
                        'supplierId': str(db_context['supplier_id']),
                        'currentStock': int(db_context['current_stock'] or 0),
                        'onOrder': int(db_context['on_order'] or 0),
                        'windowDays': 30,
                        'historyDays': history,
                        'unitsSoldInWindow': int(recent.units_sold.sum()),
                        'returnsInWindow': int(recent.returns_units.sum()),
                        'netUnitsInWindow': int(recent_net.sum()),
                        'dailyVelocity': float(recent_net.sum()) / effective_days,
                        'leadTimeDays': int(db_context['lead_time_days'] or 7),
                        'leadTimeDemand': result.lead_time_demand,
                        'safetyDays': 7,
                        'safetyStock': max(0, result.upper - result.demand),
                        'reorderPoint': result.lead_time_demand + max(0, result.upper - result.demand),
                        'reviewPeriodDays': 7,
                        'reviewPeriodDemand': result.review_period_demand,
                        'supplierName': db_context['supplier_name'],
                    }
                    upsert(connection, str(settings.tenant_id), str(store_id), str(variant_id), result, context)
                    written += 1
                summary['forecast_rows_written']=written
    except (ValueError, PermissionError, psycopg.Error) as error:
        print(f"forecast job preflight failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(summary, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
