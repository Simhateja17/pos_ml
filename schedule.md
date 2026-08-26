# Nightly schedule

On the GCP VM, `Ambel-ml.timer` invokes the one-shot `couture-forecast` job at
02:15. The service receives one restricted `ML_TENANT_ID` at a time and never
receives backend credentials. A failed run leaves the previous canonical
suggestions in place.

The temporary **Run forecast now** button inserts a row into `forecast_runs`.
`Ambel-ml-manual-worker.timer` invokes `couture-forecast-worker` every minute;
it claims at most one queued run across all tenants through the restricted
`ml_claim_next_forecast_run()` adapter, sets `app.tenant_id` from the claimed
row, records comparison items, and writes only a model that passes the
existing eligibility and WAPE gate. The worker never enumerates or directly
reads the `tenants` table.

The nightly `couture-forecast` job remains tenant-scoped and continues to
receive `ML_TENANT_ID` from its scheduler. Expanding nightly generation to
every tenant is a separate scheduling concern; it must use the same explicit
tenant enumeration boundary rather than copying IDs into a shared environment
file.
