# Nightly schedule

Run one container invocation per tenant at 02:15 local business time. A scheduler must pass `ML_TENANT_ID`; failures are caught per tenant and logged with duration and row count. The job upserts today's rows only, so a failed run leaves yesterday's suggestions untouched and checkout remains isolated.

Use a Render cron job (or equivalent) with the `ML_DATABASE_URL` secret. Never inject backend credentials.
