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
| Log Analytics workspace | `phs-logs` | Container logs (queryable via KQL) | ~free (5 GB/mo free tier) |

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
Get-Content scripts\init_db.sql -Raw          | docker compose exec -T db psql "$url"
Get-Content scripts\migrate_features.sql -Raw | docker compose exec -T db psql "$url"
Get-Content scripts\migrate_workouts.sql -Raw | docker compose exec -T db psql "$url"
Get-Content scripts\migrate_auth.sql -Raw     | docker compose exec -T db psql "$url"
Get-Content scripts\migrate_providers.sql -Raw | docker compose exec -T db psql "$url"
Get-Content scripts\migrate_sync_time.sql -Raw | docker compose exec -T db psql "$url"
Get-Content scripts\migrate_corpus.sql -Raw   | docker compose exec -T db psql "$url"
```

> `migrate_corpus.sql` retypes `chunks.embedding` to 512 dims (voyage-3.5) and adds
> the corpus metadata columns + the `source` dedup key. It nulls any existing
> 1024-dim vectors, so re-embed (or re-seed) after running it.

> Run these **in order**. `migrate_providers.sql` folds the old `whoop_connections`
> table into `provider_connections` (one row per user *and provider*, so Garmin/Oura
> slot in without new tables), copying any existing connection across and then
> dropping the old table. It's guarded, so re-running is a no-op.

> **Workouts need a WHOOP re-consent.** Reading logged workouts uses the new
> `read:workout` scope, so after deploying, the owner must **Disconnect WHOOP →
> Connect WHOOP** once to re-authorize with the added scope; existing tokens
> won't have it. Then run `migrate_workouts.sql` (above) so the table exists.

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

After seeding or re-embedding, build the ANN index once:
`Get-Content scripts\index_corpus.sql -Raw | docker compose exec -T db psql "$url"`.

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

## Observability (logs + a production canary)

Lightweight "MLOps-lite": you can see what the deployed app is doing, and a
scheduled job tells you if production quality regresses.

### Container logs → Log Analytics

The Container Apps environment was created without a log destination, so
`stdout`/`stderr` went nowhere. It now streams to a Log Analytics workspace.
How it was wired (idempotent — safe to re-run for a fresh environment):

```powershell
# 1. a workspace to hold the logs (PerGB2018; 5 GB/mo is free)
az monitor log-analytics workspace create -g phs-rg -n phs-logs `
  --location polandcentral --retention-time 30

# 2. point the Container Apps environment at it (in-place; no recreate needed)
$wsId  = az monitor log-analytics workspace show        -g phs-rg -n phs-logs --query customerId -o tsv
$wsKey = az monitor log-analytics workspace get-shared-keys -g phs-rg -n phs-logs --query primarySharedKey -o tsv
az containerapp env update -g phs-rg -n phs-env `
  --logs-destination log-analytics --logs-workspace-id $wsId --logs-workspace-key $wsKey
```

Logs begin flowing within a few minutes (the tables appear on first ingestion).
Query them with KQL — `az monitor log-analytics query` needs the workspace's
**GUID** (`customerId`), not its name:

```powershell
$wsId = az monitor log-analytics workspace show -g phs-rg -n phs-logs --query customerId -o tsv

# recent app console output (your uvicorn / print logs)
az monitor log-analytics query -w $wsId --analytics-query `
  "ContainerAppConsoleLogs_CL | where ContainerAppName_s == 'phs-api' | project TimeGenerated, Log_s | order by TimeGenerated desc | take 50" -o table

# platform/system events (restarts, scaling, image pulls)
az monitor log-analytics query -w $wsId --analytics-query `
  "ContainerAppSystemLogs_CL | where ContainerAppName_s == 'phs-api' | project TimeGenerated, Reason_s, Log_s | order by TimeGenerated desc | take 50" -o table
```

For an ad-hoc live tail without KQL, `az containerapp logs show -g phs-rg -n phs-api --follow` still works.

### Scheduled production smoke-eval

`scripts/prod_smoke_eval.py` grades the **deployed** app over HTTP — the
black-box complement to the local LangSmith golden-set harness. It's standard-
library only (no deps, no `app` imports), so it runs on a bare runner:

```powershell
python scripts/prod_smoke_eval.py                  # free: liveness + retrieval recall
python scripts/prod_smoke_eval.py --with-answers   # + /ask citation-validity (Anthropic spend)
```

`.github/workflows/prod-smoke-eval.yml` runs the **free tier daily** (06:17 UTC)
and turns the build red if the app is down or retrieval breaks. It holds **no
secrets** — the answer tier, when triggered manually via *Run workflow →
with_answers*, uses the deployed app's own Anthropic key server-side. If ingress
is ever IP-restricted (see Security notes), allow GitHub's runners or the job
can't reach the app. Point it elsewhere with an `APP_BASE_URL` repo variable.

## Accounts & auth

Real multi-user accounts: email + password (argon2id), plus Google and Microsoft
sign-in. Email verification is a **dismissible reminder, not a gate** — an
unverified account has full access, so a mail lost to spam can never lock a user
out. The flag still governs account linking, which is where it matters. Visitors
with no account can choose "Try Health Strategist now" — a guest session that
reads the sample account and writes nothing.

Sessions are server-side and revocable (`app/session.py`): the cookie holds a
random token, the database stores only its hash, and logout / password change
invalidates it. There is no longer a `SESSION_SECRET` or `OWNER_USER_ID` — the
old signed-cookie scheme and the single-owner assumption are both gone.

**Rolling this out to a fresh/existing cloud DB:**
```powershell
# 1. schema for accounts, identities, sessions, email tokens, throttling
Get-Content scripts\migrate_auth.sql -Raw | docker compose exec -T db psql "$url"
# 2. seed the sample account that guest mode reads
docker compose exec -T api python scripts/seed_demo_user.py
# 3. secrets for sign-in + email
az containerapp secret set -g phs-rg -n phs-api --secrets `
  "google-client-secret=<...>" "microsoft-client-secret=<...>" "resend-key=<...>"
az containerapp update -g phs-rg -n phs-api --set-env-vars `
  "APP_BASE_URL=https://phs-api.orangehill-97462476.polandcentral.azurecontainerapps.io" `
  "GOOGLE_CLIENT_ID=<...>" "GOOGLE_CLIENT_SECRET=secretref:google-client-secret" `
  "MICROSOFT_CLIENT_ID=<...>" "MICROSOFT_CLIENT_SECRET=secretref:microsoft-client-secret" `
  "MICROSOFT_TENANT=common" "RESEND_API_KEY=secretref:resend-key" `
  "EMAIL_FROM=Personal Health Strategist <noreply@yourdomain>"
```

**Redirect URIs** must be registered with each provider, for *both* localhost and
the live URL — Google Cloud Console and Azure AD ("Entra ID") respectively:

```
<APP_BASE_URL>/auth/oauth/google/callback
<APP_BASE_URL>/auth/oauth/microsoft/callback
```

> **Email deliverability still matters.** Verification no longer blocks access, so
> a lost mail is a nuisance rather than a lockout — but password *reset* has no
> such fallback, and Resend's shared `onboarding@resend.dev` sender only delivers
> to your own Resend account address. Verify your own sending
> domain in Resend rather than shipping with the shared test sender. With no
> `RESEND_API_KEY` set the link is written to the container logs instead, which
> is how the flow is walked locally.

## Security notes

- Every write endpoint derives the user from the session (`require_user`);
  read-only endpoints accept a guest (`readable_user`). `POST /metrics` no longer
  accepts a `user_id` in the body, `POST /users` is gone (replaced by
  `/auth/register`), and `/whoop/connect` no longer takes a `?user_id=`.
- WHOOP is **per-user**: each account connects its own, and guests cannot connect.
- Ingress may still be IP-restricted for a fully private instance:
  ```powershell
  az containerapp ingress access-restriction set -g phs-rg -n phs-api `
    --rule-name allow-my-ip --ip-address <your-ip>/32 --action Allow
  ```
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
