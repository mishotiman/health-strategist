# Deploying PHS to Azure

The live deployment runbook. Everything lives in one resource group (`phs-rg`,
Poland Central) so it can be torn down with a single command.

## What's deployed

| Resource | Name | Purpose | Rough cost |
|---|---|---|---|
| Resource group | `phs-rg` | Holds everything (the delete button) | free |
| PostgreSQL Flexible Server | `phs-db-23f1hs` | Postgres 16 + pgvector; corpus + user data | ~$12–15/mo |
| Container Registry (Basic) | `phsacr23f1hs` | Stores the app image | ~$5/mo |
| Container Apps environment | `phs-env` | Networking/logging boundary | free |
| Container App | `phs-api` | The running app + public HTTPS URL | ~free (scales to zero) |

Live URL: `https://phs-api.orangehill-97462476.polandcentral.azurecontainerapps.io`

```
docker image ──push──▶ ACR ──pull (managed identity)──▶ Container App ──▶ public HTTPS
                                                              │
                                                              ▼
                                              Postgres Flexible Server (pgvector)
```

## Prerequisites

```powershell
winget install -e --id Microsoft.AzureCLI   # then reopen the terminal
az login
az config set extension.use_dynamic_install=yes_without_prompt
```

## The deploy loop (repeat for every change)

```powershell
# 1. build   2. tag   3. push   4. roll
docker compose build api
docker tag health-strategist-api:latest phsacr23f1hs.azurecr.io/phs-api:v3
docker push phsacr23f1hs.azurecr.io/phs-api:v3
az containerapp update -g phs-rg -n phs-api --image phsacr23f1hs.azurecr.io/phs-api:v3
```

Bump the tag each time (`v3`, `v4`, …) — it makes rollback trivial:

```powershell
az containerapp update -g phs-rg -n phs-api --image phsacr23f1hs.azurecr.io/phs-api:v2
```

> **Rollouts are not instant.** The old revision keeps serving while the new one
> starts. If a fix looks missing right after `update`, wait ~60s and retest
> before debugging. Check which revision serves traffic:
> `az containerapp revision list -g phs-rg -n phs-api -o table`

## Secrets

Secrets live in the Container App's secret store, never in the image (see
`.dockerignore`, which keeps `.env` out). Env vars reference them via
`secretref:`.

```powershell
az containerapp secret list -g phs-rg -n phs-api -o table        # names only
az containerapp secret set  -g phs-rg -n phs-api --secrets anthropic-key=<new>
az containerapp update      -g phs-rg -n phs-api                 # restart to pick up
```

Currently set: `database-url`, `anthropic-key`, `voyage-key`, `whoop-id`,
`whoop-secret`, `langsmith-key`.

## Database

Connect (the local `db` container doubles as a psql client):

```powershell
$url = "postgresql://phsadmin:<password>@phs-db-23f1hs.postgres.database.azure.com:5432/phs?sslmode=require"
docker compose exec -T db psql "$url" -c "\dt"
```

Schema (already applied; re-runnable — everything is `IF NOT EXISTS`):

```powershell
Get-Content scripts\init_db.sql -Raw        | docker compose exec -T db psql "$url"
Get-Content scripts\migrate_features.sql -Raw | docker compose exec -T db psql "$url"
```

**pgvector** needs allowlisting on Azure before `CREATE EXTENSION` works:

```powershell
az postgres flexible-server parameter set -g phs-rg --server-name phs-db-23f1hs `
  --name azure.extensions --value vector
```

### Seeding data from local

The corpus embeddings are **not** in git (papers and derived text are
gitignored), so a fresh cloud DB starts empty. Copy tables from the local
container straight into Azure, then fix the id sequences:

```powershell
docker compose exec -T db sh -c "PGPASSWORD=phs pg_dump -h localhost -U phs -d phs --data-only -t documents -t chunks | psql '$url'"
docker compose exec -T db psql "$url" -c "SELECT setval('documents_id_seq',(SELECT MAX(id) FROM documents)); SELECT setval('chunks_id_seq',(SELECT MAX(id) FROM chunks));"
```

Same pattern for `users`, `profiles`, `health_metrics`. Alternative: point the
ingestion scripts at the cloud `DATABASE_URL` and re-embed (costs Voyage calls).

### Firewall

```powershell
az postgres flexible-server firewall-rule list -g phs-rg --server-name phs-db-23f1hs -o table
# your home IP changes? re-add it:
az postgres flexible-server firewall-rule create -g phs-rg --server-name phs-db-23f1hs `
  --name my-laptop --start-ip-address <ip> --end-ip-address <ip>
```

`allow-azure-services` (0.0.0.0–0.0.0.0) is what lets the Container App connect.

## WHOOP

`WHOOP_REDIRECT_URI` must match a redirect URI registered in the WHOOP developer
app, exactly:

```
https://phs-api.orangehill-97462476.polandcentral.azurecontainerapps.io/whoop/callback
```

`/whoop/connect?user_id=N` returns **404** if user N doesn't exist — onboard
first via `POST /users`.

## Cost controls

```powershell
# Pause the DB (the main cost). NOTE: Azure auto-restarts it after 7 days.
az postgres flexible-server stop  -g phs-rg -n phs-db-23f1hs
az postgres flexible-server start -g phs-rg -n phs-db-23f1hs

# Keep the app warm for a demo (~$14/mo) vs scale-to-zero (free when idle, ~20-60s cold start)
az containerapp update -g phs-rg -n phs-api --min-replicas 1
az containerapp update -g phs-rg -n phs-api --min-replicas 0

# Nuclear: delete everything, all charges stop
az group delete --name phs-rg --yes --no-wait
```

ACR Basic (~$5/mo) has no stop — only delete.

## Auth / demo user

The public URL defaults to a synthetic **Demo User**; real data is behind a
login. Completing WHOOP OAuth sets a signed session cookie (`app/session.py`)
identifying the browser as the owner account; without it, callers are the demo
user. `/chat`, `/whoop/sync`, and `/metrics/{id}` derive the user from the
session, so real data (`/metrics/1`) returns 403 without the cookie.

**Rolling this out to a fresh/existing cloud DB:**
```powershell
# 1. migration adds users.display_name + users.is_demo
Get-Content scripts\migrate_features.sql -Raw | docker compose exec -T db psql "$url"
# 2. seed the demo user (run against the cloud DB: point the api container's
#    DATABASE_URL at $url, or run the SQL/seed with that connection)
docker compose exec -T api python scripts/seed_demo_user.py
# 3. set a real session secret (REQUIRED in prod — dev default is insecure)
az containerapp secret set -g phs-rg -n phs-api --secrets "session-secret=<long-random>"
az containerapp update -g phs-rg -n phs-api `
  --set-env-vars "SESSION_SECRET=secretref:session-secret" "OWNER_USER_ID=1"
```
After this, `SESSION_SECRET` joins the secret list below. Confirm the WHOOP
callback URL is registered (it already is) so login redirects work.

## Security notes

- **Auth is demo-user + WHOOP-login** (above). Ingress no longer *needs* to be
  IP-restricted to protect data, but you can still lock it to your IP for a
  fully private instance:
  ```powershell
  az containerapp ingress access-restriction set -g phs-rg -n phs-api `
    --rule-name allow-my-ip --ip-address <your-ip>/32 --action Allow
  ```
- Write endpoints (`POST /metrics`, `/upload/bloodwork`, `POST /users`) still
  take a `user_id` and are not yet session-gated — a follow-up if needed.
- ACR admin user is disabled; the Container App pulls via its system-assigned
  managed identity.
- `MemorySaver` keeps agent conversation state in RAM, so keep `--max-replicas 1`
  (or move to a Postgres checkpointer) — otherwise memory is inconsistent
  across instances.

## Gotchas hit during setup

- **West Europe / Germany West Central are restricted** for new subscriptions
  ("The location is restricted from performing this operation"). Poland Central
  works and is closest to Sofia. Check a region before committing:
  `az postgres flexible-server list-skus --location <loc> -o json`
- `--public-access None` at server-create time **disables public networking
  entirely**, which then blocks adding firewall rules. Fix:
  `az postgres flexible-server update -g phs-rg -n phs-db-23f1hs --public-access Enabled`
- CLI flag naming: it's `--name` for the database/firewall-rule name and
  `--server-name` for the server (not `--database-name` / `--rule-name`).
