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

    with psycopg.connect(settings.database_url) as connection:
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
        summary = inspect_rollup(Settings.from_environment())
    except (ValueError, PermissionError, psycopg.Error) as error:
        print(f"forecast job preflight failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(summary, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
