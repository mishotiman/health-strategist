"""Ingestion — chunk the plain-text corpus into the documents + chunks tables.

    docker compose exec -T api python scripts/ingest.py

Reads data/text/*.txt (produced by extract_text.py) and data/text/_metadata.json,
splits each paper into recursive/structure-aware chunks, and loads them.
Embeddings are left NULL here — filled later by embed_chunks.py. Re-running is
incremental and idempotent: a paper already loaded (matched by its `source`) is
skipped, so only new papers are chunked — safe to run repeatedly as the corpus grows.
"""

from __future__ import annotations

import glob
import json
import os
import re

import psycopg

TEXT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "text")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://phs:phs@db:5432/phs")

MAX_CHARS = 1000       # ~250 tokens
OVERLAP_CHARS = 150    # ~15% overlap


def recursive_split(text: str, max_chars: int = MAX_CHARS,
                    overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split on the largest natural boundary that keeps pieces <= max_chars,
    descending paragraph -> line -> sentence -> word -> hard cut, then add a
    character overlap between consecutive pieces so context isn't lost at seams.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(s: str, seps: list[str]) -> list[str]:
        if len(s) <= max_chars:
            return [s]
        sep = seps[0]
        if sep == "":  # last resort: hard character cut
            return [s[i:i + max_chars] for i in range(0, len(s), max_chars)]
        pieces: list[str] = []
        current = ""
        for part in s.split(sep):
            candidate = part if not current else current + sep + part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                if len(part) > max_chars:
                    pieces.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if current:
            pieces.append(current)
        return pieces

    raw = [p.strip() for p in _split(text, separators) if p.strip()]
    if overlap <= 0 or len(raw) <= 1:
        return raw
    out = [raw[0]]
    for i in range(1, len(raw)):
        out.append(raw[i - 1][-overlap:] + " " + raw[i])
    return out


def parse_txt(path: str) -> list[dict]:
    """Turn a plain-text paper into blocks {heading, text, page}, using the
    "## heading" / "## Page N" marker lines written by extract_text.py."""
    blocks: list[dict] = []
    heading, page, buffer = "", None, []

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            blocks.append({"heading": heading, "text": body, "page": page})
        buffer = []

    with open(path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            if line.startswith("## "):
                flush()
                label = line[3:].strip()
                page_marker = re.match(r"Page (\d+)$", label)
                if page_marker:
                    heading, page = "", int(page_marker.group(1))
                else:
                    heading, page = label, None
            elif line.startswith("# "):  # document title line — skip
                continue
            else:
                buffer.append(line)
    flush()
    return blocks


def blocks_to_chunks(blocks: list[dict]) -> list[dict]:
    """Chunk each block, prefixing its section heading so the passage carries
    a little context."""
    chunks: list[dict] = []
    for block in blocks:
        for piece in recursive_split(block["text"]):
            content = f"{block['heading']}\n\n{piece}" if block["heading"] else piece
            chunks.append({"content": content, "page": block["page"]})
    return chunks


def main() -> None:
    meta_path = os.path.join(TEXT_DIR, "_metadata.json")
    if not os.path.exists(meta_path):
        raise SystemExit("data/text/_metadata.json not found — run extract_text.py first.")
    metadata = json.load(open(meta_path, encoding="utf-8"))

    paths = sorted(glob.glob(os.path.join(TEXT_DIR, "*.txt")))
    if not paths:
        raise SystemExit(f"No text files in {TEXT_DIR} — run extract_text.py first.")

    loaded = skipped = total_chunks = 0
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for path in paths:
                key = os.path.basename(path)
                meta = metadata.get(key, {"title": key, "authors": "",
                                          "year": None, "source": ""})
                # Idempotent by `source` (the natural key from migrate_corpus.sql):
                # a paper already loaded returns no row, so we skip re-chunking it.
                cur.execute(
                    "INSERT INTO documents "
                    "  (title, source, authors, year, pmcid, doi, license, pillar) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (source) DO NOTHING RETURNING id",
                    (meta["title"], meta["source"], meta["authors"], meta["year"],
                     meta.get("pmcid"), meta.get("doi"), meta.get("license"),
                     meta.get("pillar")),
                )
                row = cur.fetchone()
                if row is None:
                    skipped += 1
                    continue
                doc_id = row[0]

                chunks = blocks_to_chunks(parse_txt(path))
                cur.executemany(
                    "INSERT INTO chunks (document_id, chunk_index, content, page) "
                    "VALUES (%s, %s, %s, %s)",
                    [(doc_id, idx, ch["content"], ch["page"])
                     for idx, ch in enumerate(chunks)],
                )
                loaded += 1
                total_chunks += len(chunks)
                print(f"  {key:48} -> {len(chunks):4} chunks")

        conn.commit()

    print(f"\nDone: {loaded} new documents ({total_chunks} chunks); "
          f"{skipped} already present.")


if __name__ == "__main__":
    main()
