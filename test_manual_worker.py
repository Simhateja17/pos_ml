import os
import unittest
from decimal import Decimal
from unittest.mock import patch

import pandas as pd

import manual_worker
from forecast import daily_series
from job import _connect


class _Cursor:
    def __init__(self, row):
        self.row = row
        self.query = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params=None):
        self.query = query
        self.params = params

    def fetchone(self):
        return self.row


class _Connection:
    def __init__(self, row):
        self.cursor_instance = _Cursor(row)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def transaction(self):
        return self

    def cursor(self, row_factory=None):
        return self.cursor_instance


class ManualWorkerTests(unittest.TestCase):
    def test_eligibility_metrics_count_calendar_days_and_recent_calendar_window(self):
        rows = pd.DataFrame(
            {
                "variant_id": ["variant"] * 3,
                "date": ["2026-06-01", "2026-07-01", "2026-08-20"],
                "units_sold": [20, 20, 5],
                "returns_units": [0, 0, 1],
            }
        )

        history_days, trailing_units, total_units = manual_worker._eligibility_metrics(
            rows, "2026-08-26"
        )

        self.assertEqual(history_days, 81)
        self.assertEqual(trailing_units, 4)
        self.assertEqual(total_units, 44)

    def test_eligibility_metrics_do_not_treat_old_active_rows_as_recent_days(self):
        rows = pd.DataFrame(
            {
                "variant_id": ["variant"] * 2,
                "date": ["2026-06-01", "2026-08-01"],
                "units_sold": [30, 5],
                "returns_units": [0, 0],
            }
        )

        history_days, trailing_units, total_units = manual_worker._eligibility_metrics(
            rows, "2026-08-26"
        )

        self.assertEqual(history_days, 62)
        self.assertEqual(trailing_units, 0)
        self.assertEqual(total_units, 35)

    def test_heuristic_uses_calendar_history_and_recent_window(self):
        rows = pd.DataFrame(
            {
                "variant_id": ["variant"] * 3,
                "date": ["2026-06-01", "2026-07-01", "2026-08-20"],
                "units_sold": [20, 20, 5],
                "returns_units": [0, 0, 1],
            }
        )

        result = manual_worker._heuristic(rows, {}, pd.Timestamp("2026-08-26"))

        self.assertEqual(result["historyDays"], 81)
        self.assertEqual(result["unitsSoldInWindow"], 5)
        self.assertEqual(result["returnsInWindow"], 1)
        self.assertEqual(result["netUnitsInWindow"], 4)

    def test_daily_series_coerces_database_numeric_values(self):
        rows = pd.DataFrame(
            {
                "variant_id": ["variant", "variant"],
                "date": ["2026-06-01", "2026-06-03"],
                "units_sold": [Decimal("2.5"), Decimal("4")],
                "returns_units": [Decimal("0"), Decimal("1")],
            }
        )

        series = daily_series(rows, {pd.Timestamp("2026-06-02")})

        self.assertEqual(series["y"].dtype.kind, "f")
        self.assertEqual(len(series), 3)

    def test_claim_uses_global_queue_rpc_without_tenant_parameter(self):
        row = {
            "run_id": "run",
            "tenant_id": "tenant-from-queue",
            "store_id": "store",
        }
        connection = _Connection(row)

        claimed = manual_worker._claim(connection)

        self.assertEqual(claimed, row)
        self.assertEqual(connection.cursor_instance.query, "select * from public.ml_claim_next_forecast_run()")
        self.assertIsNone(connection.cursor_instance.params)

    def test_worker_settings_does_not_require_single_tenant_id(self):
        with patch.dict(
            os.environ,
            {"ML_DATABASE_URL": "postgresql://user:password@example.test:5432/postgres", "ML_TENANT_ID": ""},
            clear=False,
        ):
            settings = manual_worker.WorkerSettings.from_environment()

        self.assertEqual(settings.database_url, "postgresql://user:password@example.test:5432/postgres")

    def test_ml_connection_disables_prepared_statements_for_transaction_pooling(self):
        class _Connection:
            prepare_threshold = None

        with patch("job.psycopg.connect", return_value=_Connection()) as connect:
            connection = _connect("postgresql://user:password@example.test:6543/postgres")

        self.assertIsNotNone(connection)
        connect.assert_called_once_with(
            "postgresql://user:password@example.test:6543/postgres",
            prepare_threshold=None,
        )


if __name__ == "__main__":
    unittest.main()
