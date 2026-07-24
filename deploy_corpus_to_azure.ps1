# One-off: seed the Azure production DB with your local corpus, then index it.
# YOU run this — it prompts for your Azure DB password (never leaves your machine).
# Keep the laptop awake ~15-20 min (the ~2GB copy is the slow part).
# Firewall for your IP is already in place. Delete this file when done.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot   # run from wherever the repo lives

function Must($step) {
  if ($LASTEXITCODE -ne 0) { Write-Host "`nFAILED at: $step (exit $LASTEXITCODE). Fix, then re-run." -ForegroundColor Red; exit 1 }
}

$sec = Read-Host -AsSecureString 'Azure DB password (phsadmin)'
$pw  = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
         [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
$enc = [uri]::EscapeDataString($pw)
$url = "postgresql://phsadmin:$enc@phs-db-23f1hs.postgres.database.azure.com:5432/phs?sslmode=require"

Write-Host "`n[1/4] Migrating schema (vector(512) + columns + dedup key) ..." -ForegroundColor Cyan
Get-Content scripts\migrate_corpus.sql -Raw | docker compose exec -T db psql "$url"; Must 'migrate schema'

Write-Host "`n[2/4] Replacing corpus: truncate + copy ~2GB (the slow part) ..." -ForegroundColor Cyan
docker compose exec -T db psql "$url" -c "TRUNCATE chunks, documents RESTART IDENTITY CASCADE;"; Must 'truncate'
docker compose exec -T db sh -c "PGPASSWORD=phs pg_dump -h localhost -U phs -d phs --data-only -t documents -t chunks | psql '$url'"; Must 'copy documents+chunks'
docker compose exec -T db psql "$url" -c "SELECT setval('documents_id_seq',(SELECT MAX(id) FROM documents)); SELECT setval('chunks_id_seq',(SELECT MAX(id) FROM chunks));"; Must 'fix sequences'

Write-Host "`n[3/4] Building the HNSW vector index ..." -ForegroundColor Cyan
Get-Content scripts\index_corpus.sql -Raw | docker compose exec -T db psql "$url"; Must 'build index'

Write-Host "`n[4/4] Verifying ..." -ForegroundColor Cyan
docker compose exec -T db psql "$url" -c "SELECT count(*) AS documents FROM documents;"
docker compose exec -T db psql "$url" -c "SELECT count(embedding) AS embedded, count(*) FILTER (WHERE embedding IS NULL) AS unembedded FROM chunks;"
docker compose exec -T db psql "$url" -c "SELECT indisvalid AS index_valid FROM pg_index WHERE indexrelid='ix_chunks_embedding_hnsw'::regclass;"

Write-Host "`nDB loaded. Tell Claude 'DB is loaded' and it will deploy the v7 app image." -ForegroundColor Green
