import os
import unittest
from unittest.mock import patch

import manual_worker
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
