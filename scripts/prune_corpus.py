"""Prune fetched corpus papers that violate the current recipe exclusions.

    docker compose exec -T api python scripts/prune_corpus.py            # preview
    docker compose exec -T api python scripts/prune_corpus.py --apply    # delete

Re-vets every fetched paper (those with a pillar tag; the original hand-picked
sample, pillar NULL, is left alone) against data/corpus_topics.yml's `exclude`
terms, matched against the title. Useful after tightening exclusions: papers
fetched under a looser recipe (e.g. the initial smoke slice, taken before the
exclusions existed) get removed from the DB, disk, and the corpus index so
retrieval — and any later re-ingest — stay consistent. Matching is title-only
(abstracts aren't stored), which catches the clear violations.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import psycopg
import yaml

HERE = os.path.dirname(__file__)
PAPERS_DIR = os.path.join(HERE, "..", "data", "papers")
TEXT_DIR = os.path.join(HERE, "..", "data", "text")
INDEX_PATH = os.path.join(PAPERS_DIR, "_corpus_index.json")
TOPICS_PATH = os.path.join(HERE, "..", "data", "corpus_topics.yml")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://phs:phs@db:5432/phs")


def exclusion_terms() -> list[str]:
    """Every exclude term in the recipe (global defaults + per-pillar), deduped."""
    topics = yaml.safe_load(open(TOPICS_PATH, encoding="utf-8"))
    clauses = [topics.get("defaults", {}).get("exclude", "")]
    clauses += [p.get("exclude", "") for p in topics["pillars"]]
    terms: list[str] = []
    for clause in clauses:
        for t in clause.split(" OR "):
            t = t.strip().strip('"').strip()
            if t and t not in terms:
                terms.append(t)
    return terms


def build_pattern(terms: list[str]) -> str:
    # Word-boundaried alternation so "cognition" doesn't match "recognition".
    return r"\y(" + "|".join(re.escape(t) for t in terms) + r")\y"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually delete the matches (default: preview only)")
    args = ap.parse_args()

    pattern = build_pattern(exclusion_terms())

    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, pmcid, pillar, title FROM documents "
            "WHERE pillar IS NOT NULL AND title ~* %s ORDER BY pillar, id",
            (pattern,),
        )
        rows = cur.fetchall()

        if not rows:
            print("Nothing to prune — no fetched paper's title hits an exclusion term.")
            return

        by_pillar: dict[str, int] = {}
        for _id, _pmcid, pillar, _title in rows:
            by_pillar[pillar] = by_pillar.get(pillar, 0) + 1
        print(f"{len(rows)} fetched papers match a current exclusion term (by title):")
        for pil, n in sorted(by_pillar.items(), key=lambda kv: -kv[1]):
            print(f"  {pil:22} {n}")
        print("\n  examples:")
        for _id, _pmcid, pillar, title in rows[:15]:
            print(f"   [{pillar}] {(title or '')[:74]}")

        if not args.apply:
            print("\nPreview only. Re-run with --apply to delete from DB + disk + index.")
            return

        ids = [r[0] for r in rows]
        cur.execute("DELETE FROM documents WHERE id = ANY(%s)", (ids,))  # cascades to chunks
        conn.commit()

    # Remove on-disk artifacts so a later re-ingest can't resurrect them.
    pmcids = [r[1] for r in rows if r[1]]
    removed_files = 0
    for pmcid in pmcids:
        for path in (os.path.join(PAPERS_DIR, f"{pmcid}.xml"),
                     os.path.join(TEXT_DIR, f"{pmcid}.txt")):
            if os.path.exists(path):
                os.remove(path)
                removed_files += 1

    # Prune the corpus index sidecar.
    if os.path.exists(INDEX_PATH):
        index = json.load(open(INDEX_PATH, encoding="utf-8"))
        drop = {f"{p}.xml" for p in pmcids}
        index = {k: v for k, v in index.items() if k not in drop}
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nPruned {len(ids)} documents (+ their chunks), removed {removed_files} files, "
          f"and updated the corpus index.")


if __name__ == "__main__":
    main()
