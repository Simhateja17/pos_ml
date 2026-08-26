# Couture POS forecasting job

This repository is the isolated Python batch boundary for ML-02. It reads
tenant-scoped rows from `daily_sales_rollup` and, in later Phase 6 tasks, writes
forecast-backed rows to `reorder_suggestions`. The Express application never
calls Python, and this repository cannot import code from `backend/`.

[Nixtla `statsforecast`](https://github.com/Nixtla/statsforecast) is the
forecasting engine. Prophet is intentionally not used.

## Database boundary

Migration `0027_ml_forecast_role.sql` creates the `ml_forecast` Postgres role
with exactly:

- `SELECT` on `daily_sales_rollup`
- `INSERT`, `UPDATE` on `reorder_suggestions`

The role is `NOINHERIT` and `NOBYPASSRLS`. It cannot read `sales`, `customers`,
or any other application table. Do not put a superuser URL or the
`app_runtime` URL in `ML_DATABASE_URL`.

The migration does not commit a password. Provision or rotate the role password
through the deployment secret manager, then configure the two values shown in
`.env.example`.

## Local preflight

Python 3.11 or newer is required.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
ML_DATABASE_URL='postgresql://...' \
ML_TENANT_ID='tenant-uuid' \
.venv/bin/couture-forecast
```

The temporary manual-test queue uses the same restricted connection through a
separate, one-shot worker. It claims one queued run across all tenants through
a narrow database adapter, then sets the returned tenant as its RLS context:

```sh
ML_DATABASE_URL='postgresql://...' \
.venv/bin/couture-forecast-worker
```

The worker records a side-by-side heuristic/ML comparison and writes only
genuine winning forecast suggestions to `reorder_suggestions`. It has no HTTP
surface. On the GCP VM it is invoked by `Ambel-ml-manual-worker.timer`.

Task 1's command is intentionally only a preflight. It imports the pinned
forecasting stack, sets the tenant RLS context, reads the tenant's rollup, and
fails closed if the connected role has any table privilege outside the two-table
allow-list. Modeling and write-back arrive in Tasks 2–4.
