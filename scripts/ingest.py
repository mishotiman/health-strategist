"""Ingestion v0 — parse the seed corpus into the documents + chunks tables.

Run inside the api container so the `db` hostname resolves:

    docker compose exec -T api python scripts/ingest.py

Parses full-text JATS XML (Europe PMC) and one PDF (SportRxiv), splits each
paper into recursive/structure-aware chunks, and loads them. Embeddings are
left NULL here — they get filled on Wednesday. Re-running is idempotent
(truncate + reload).
"""

from __future__ import annotations

import glob
import os
import re
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF
import psycopg

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "papers")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://phs:phs@db:5432/phs")

MAX_CHARS = 1000       # ~250 tokens
OVERLAP_CHARS = 150    # ~15% overlap


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def _localname(tag: str) -> str:
    return tag.split("}")[-1]


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def recursive_split(text: str, max_chars: int = MAX_CHARS,
                    overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split on the largest natural boundary that keeps pieces <= max_chars,
    descending paragraph -> line -> sentence -> word -> hard cut. Then add a
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


# --------------------------------------------------------------------------- #
# Parsers -> list of blocks {heading, text, page}
# --------------------------------------------------------------------------- #
def parse_jats(path: str) -> dict:
    root = ET.parse(path).getroot()
    front = root.find(".//front")

    title = _text(front.find(".//title-group/article-title")) if front is not None else ""
    journal = _text(front.find(".//journal-title")) if front is not None else ""

    authors: list[str] = []
    if front is not None:
        for contrib in front.findall(".//contrib"):
            # Some journals tag <contrib contrib-type="author">; others leave
            # <contrib> untyped inside <contrib-group content-type="author">.
            # Accept authors (or untyped) and skip editors / other roles.
            ctype = contrib.get("contrib-type")
            if ctype and ctype != "author":
                continue
            surname = _text(contrib.find(".//surname"))
            given = _text(contrib.find(".//given-names"))
            name = " ".join(x for x in [given, surname] if x)
            if name:
                authors.append(name)

    year = None
    if front is not None:
        for y in front.findall(".//pub-date/year"):
            if _text(y).isdigit():
                year = int(_text(y))
                break

    doi = _text(front.find('.//article-id[@pub-id-type="doi"]')) if front is not None else ""
    pmcid_match = re.search(r"(PMC\d+)", os.path.basename(path))
    pmcid = pmcid_match.group(1) if pmcid_match else ""
    source = (f"https://doi.org/{doi}" if doi
              else f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/")

    blocks: list[dict] = []
    body = root.find(".//body")
    if body is not None:
        # paragraphs sitting directly under <body> (no section)
        for p in body.findall("p"):
            t = _text(p)
            if t:
                blocks.append({"heading": "", "text": t, "page": None})
        # one block per section (direct child <p>s only, so nested subsections
        # become their own blocks in document order — no duplication)
        for sec in body.findall(".//sec"):
            heading = _text(sec.find("title"))
            paras = [_text(p) for p in sec.findall("p")]
            sec_text = "\n\n".join(p for p in paras if p)
            if sec_text:
                blocks.append({"heading": heading, "text": sec_text, "page": None})

    return {"title": title, "authors": ", ".join(authors), "year": year,
            "source": source, "blocks": blocks}


def parse_pdf(path: str) -> dict:
    doc = fitz.open(path)
    blocks: list[dict] = []
    for page_no, page in enumerate(doc, start=1):
        text = re.sub(r"[ \t]+", " ", page.get_text()).strip()
        if text:
            blocks.append({"heading": "", "text": text, "page": page_no})
    doc.close()

    # Metadata for the one PDF (SportRxiv preprint) comes from the manifest.
    title = doc.metadata.get("title") or os.path.basename(path)
    return {"title": "The Resistance Training Dose-Response (Pelland et al., 2024)",
            "authors": "Pelland, Remmert, Robinson, Hinson, Zourdos", "year": 2024,
            "source": "https://sportrxiv.org/index.php/server/preprint/view/460",
            "blocks": blocks}


# --------------------------------------------------------------------------- #
# Chunk + load
# --------------------------------------------------------------------------- #
def blocks_to_chunks(blocks: list[dict]) -> list[dict]:
    """Turn parsed blocks into stored chunks, prefixing each with its section
    heading so the passage carries a little context."""
    chunks: list[dict] = []
    for block in blocks:
        for piece in recursive_split(block["text"]):
            content = f"{block['heading']}\n\n{piece}" if block["heading"] else piece
            chunks.append({"content": content, "page": block["page"]})
    return chunks


def main() -> None:
    paths = sorted(glob.glob(os.path.join(PAPERS_DIR, "*.xml")) +
                   glob.glob(os.path.join(PAPERS_DIR, "*.pdf")))
    if not paths:
        raise SystemExit(f"No papers found in {PAPERS_DIR}")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE;")

            total_chunks = 0
            for path in paths:
                parsed = parse_pdf(path) if path.endswith(".pdf") else parse_jats(path)
                cur.execute(
                    "INSERT INTO documents (title, source, authors, year) "
                    "VALUES (%s, %s, %s, %s) RETURNING id",
                    (parsed["title"], parsed["source"], parsed["authors"], parsed["year"]),
                )
                doc_id = cur.fetchone()[0]

                chunks = blocks_to_chunks(parsed["blocks"])
                for idx, ch in enumerate(chunks):
                    cur.execute(
                        "INSERT INTO chunks (document_id, chunk_index, content, page) "
                        "VALUES (%s, %s, %s, %s)",
                        (doc_id, idx, ch["content"], ch["page"]),
                    )
                total_chunks += len(chunks)
                print(f"  {os.path.basename(path):48} -> {len(chunks):4} chunks")

        conn.commit()

    print(f"\nDone: {len(paths)} documents, {total_chunks} chunks loaded.")


if __name__ == "__main__":
    main()
