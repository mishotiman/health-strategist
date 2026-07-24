"""Acquire an open-access research corpus from Europe PMC per data/corpus_topics.yml.

    docker compose exec -T api python scripts/fetch_corpus.py            # full run
    docker compose exec -T api python scripts/fetch_corpus.py --slice 15 # smoke test
    docker compose exec -T api python scripts/fetch_corpus.py --dry-run  # preview, no download

For each pillar's queries this searches the Europe PMC open-access subset (biased
toward high-evidence synthesis publication types and recency, and filtered by the
recipe's `exclude` terms so the corpus stays adult + physical), then downloads up
to the pillar's `cap` full-text JATS XML papers into data/papers/<PMCID>.xml. It
records {pmcid, doi, license, pillar} for each paper in data/papers/_corpus_index.json
so the extraction step can tag documents.

Idempotent: papers already on disk (and already claimed by an earlier pillar) are
skipped, so a re-run only fetches what's new and the index accumulates. `--slice N`
caps every pillar to N papers for a quick end-to-end check before the full run.
`--dry-run` searches and lists the candidates per pillar (full list ->
data/papers/_corpus_preview.json) without downloading — use it to vet the selection
before committing to the fetch.

Only JATS XML from the OA subset is handled here — that's the format from_jats()
in extract_text.py already parses.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import yaml
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

HERE = os.path.dirname(__file__)
PAPERS_DIR = os.path.join(HERE, "..", "data", "papers")
TOPICS_PATH = os.path.join(HERE, "..", "data", "corpus_topics.yml")
INDEX_PATH = os.path.join(PAPERS_DIR, "_corpus_index.json")

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# Full text is keyed by the PMCID directly (e.g. .../rest/PMC1234567/fullTextXML) —
# NOT under a /PMC/ source segment, and the id keeps its "PMC" prefix.
FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

PAGE_SIZE = 100
REQUEST_DELAY = 0.34          # be a polite client: ~3 requests/second
USER_AGENT = "PHS-corpus-fetcher/1.0 (research use)"


def _is_transient(exc: BaseException) -> bool:
    """Retry only on server/rate-limit/network errors — never on a 404 (a paper
    simply having no OA full text is not worth four retries)."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, urllib.error.URLError)  # DNS / connection / timeout


@retry(retry=retry_if_exception(_is_transient),
       stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=2, max=30))
def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def build_query(topic: str, year_min: int, pub_types: list[str], with_pub_types: bool,
                exclude: str = "") -> str:
    """AND the pillar's topic query with the OA/recency filters (and, on the first
    pass, the high-evidence publication-type preference), then subtract the
    exclusion terms that keep the corpus adult + physical + high-evidence."""
    parts = [f"({topic})", "OPEN_ACCESS:Y", "IN_EPMC:Y", f"PUB_YEAR:[{year_min} TO *]"]
    if with_pub_types and pub_types:
        types = " OR ".join(f'PUB_TYPE:"{t}"' for t in pub_types)
        parts.append(f"({types})")
    query = " AND ".join(parts)
    if exclude:
        query += f" AND NOT ({exclude})"
    return query


def search(query: str, limit: int) -> list[dict]:
    """Page the Europe PMC search API and return up to `limit` OA hits, each a
    {pmcid, doi, license, title}. Uses cursorMark paging (stable for deep result
    sets)."""
    hits: list[dict] = []
    cursor = "*"
    while len(hits) < limit:
        params = urllib.parse.urlencode({
            "query": query, "resultType": "core", "format": "json",
            "pageSize": PAGE_SIZE, "cursorMark": cursor,
        })
        payload = json.loads(_get(f"{SEARCH_URL}?{params}"))
        results = payload.get("resultList", {}).get("result", [])
        if not results:
            break
        for r in results:
            pmcid = r.get("pmcid")
            if not pmcid or r.get("isOpenAccess") != "Y":
                continue                       # need a PMCID + OA to fetch full text
            hits.append({"pmcid": pmcid, "doi": r.get("doi", ""),
                         "license": r.get("license", ""), "title": r.get("title", "")})
            if len(hits) >= limit:
                break
        next_cursor = payload.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break                              # last page
        cursor = next_cursor
        time.sleep(REQUEST_DELAY)
    return hits


def download_fulltext(pmcid: str) -> bool:
    """Fetch a paper's JATS full text to data/papers/<PMCID>.xml. Returns False if
    it's already on disk-less (no OA full text) — the caller then skips it."""
    path = os.path.join(PAPERS_DIR, f"{pmcid}.xml")
    if os.path.exists(path):
        return True
    try:
        xml = _get(FULLTEXT_URL.format(pmcid=pmcid))
    except urllib.error.HTTPError as e:
        print(f"    ! {pmcid}: no full text ({e.code})")
        return False
    if b"<article" not in xml[:8000]:          # guard against error/empty responses
        print(f"    ! {pmcid}: response is not JATS XML, skipping")
        return False
    with open(path, "wb") as f:
        f.write(xml)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=0,
                    help="cap every pillar to N papers (quick smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="search + list candidate papers per pillar; download nothing "
                         "and touch neither the corpus nor its index")
    args = ap.parse_args()
    dry = args.dry_run

    os.makedirs(PAPERS_DIR, exist_ok=True)
    topics = yaml.safe_load(open(TOPICS_PATH, encoding="utf-8"))
    defaults = topics.get("defaults", {})
    default_year = defaults.get("year_min", 2015)
    pub_types = defaults.get("pub_types", [])
    default_exclude = defaults.get("exclude", "")

    # A real run resumes from the existing index; a dry run starts clean so it
    # shows the complete recipe selection, not just what's still missing.
    index: dict = {} if dry else (
        json.load(open(INDEX_PATH, encoding="utf-8")) if os.path.exists(INDEX_PATH) else {})
    claimed_pmcids = {v["pmcid"] for v in index.values()}
    claimed_dois = {v["doi"] for v in index.values() if v.get("doi")}
    preview: list[dict] = []   # dry-run only

    def save_index() -> None:
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    total = 0
    for pillar in topics["pillars"]:
        name = pillar["name"]
        cap = args.slice or pillar["cap"]
        year_min = pillar.get("year_min", default_year)
        exclude = default_exclude
        if pillar.get("exclude"):
            exclude = f"{exclude} OR {pillar['exclude']}" if exclude else pillar["exclude"]
        print(f"\n== {name}  (target {cap}) ==")

        added, samples = 0, []
        # First pass prefers high-evidence syntheses; if that leaves the pillar
        # under-filled (a niche topic), a second pass drops the pub-type filter.
        for with_pub_types in (True, False):
            if added >= cap:
                break
            for query in pillar["queries"]:
                if added >= cap:
                    break
                # real runs over-fetch to absorb dedup + no-full-text attrition;
                # a dry run has no downloads, so `cap` candidates is enough.
                limit = cap if dry else cap * 3
                for hit in search(build_query(query, year_min, pub_types,
                                              with_pub_types, exclude), limit):
                    if added >= cap:
                        break
                    pmcid, doi = hit["pmcid"], hit.get("doi", "")
                    if pmcid in claimed_pmcids or (doi and doi in claimed_dois):
                        continue
                    if not dry and not download_fulltext(pmcid):
                        continue
                    claimed_pmcids.add(pmcid)
                    if doi:
                        claimed_dois.add(doi)
                    if dry:
                        preview.append({"pmcid": pmcid, "doi": doi,
                                        "license": hit.get("license", ""),
                                        "pillar": name, "title": hit.get("title", "")})
                    else:
                        index[f"{pmcid}.xml"] = {"pmcid": pmcid, "doi": doi,
                                                 "license": hit.get("license", ""), "pillar": name}
                    if len(samples) < 5:
                        samples.append((hit.get("title", "") or "")[:80])
                    added += 1
                    total += 1
                    if not dry:
                        time.sleep(REQUEST_DELAY)
                if not dry:
                    save_index()               # persist after each query, crash-safe
        print(f"   {name}: {added} candidate(s)" if dry else f"   {name}: +{added} papers")
        for t in samples if dry else []:
            print(f"      - {t}")

    if dry:
        preview_path = os.path.join(PAPERS_DIR, "_corpus_preview.json")
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(preview, f, ensure_ascii=False, indent=2)
        print(f"\nDry run: {total} candidate papers across {len(topics['pillars'])} pillars.")
        print(f"Full list (title + PMCID + pillar): {preview_path}. Nothing downloaded.")
    else:
        save_index()
        print(f"\nDone: {total} new papers; {len(index)} in the corpus index.")


if __name__ == "__main__":
    main()
