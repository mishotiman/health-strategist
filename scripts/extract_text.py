"""Extract raw papers (JATS XML + PDF) to clean plain text + a metadata sidecar.

    docker compose exec -T api python scripts/extract_text.py

Writes one data/text/<name>.txt per source paper and data/text/_metadata.json
with the citation fields (title, authors, year, source). Section headings are
emitted as "## <heading>" marker lines (or "## Page N" for the PDF) so the
downstream chunker can keep section context while the file stays human-readable.

This decouples messy format parsing from chunking: ingest.py reads only the
plain-text files.
"""

from __future__ import annotations

import glob
import json
import os
import re
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "papers")
TEXT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "text")
CORPUS_INDEX = os.path.join(PAPERS_DIR, "_corpus_index.json")  # written by fetch_corpus.py


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def from_jats(path: str) -> tuple[dict, str]:
    root = ET.parse(path).getroot()
    front = root.find(".//front")

    title = _text(front.find(".//title-group/article-title")) if front is not None else ""

    authors: list[str] = []
    if front is not None:
        for contrib in front.findall(".//contrib"):
            ctype = contrib.get("contrib-type")
            if ctype and ctype != "author":  # skip editors / other roles
                continue
            name = " ".join(x for x in [_text(contrib.find(".//given-names")),
                                        _text(contrib.find(".//surname"))] if x)
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
    meta = {"title": title, "authors": ", ".join(authors), "year": year, "source": source,
            "pmcid": pmcid, "doi": doi}

    out: list[str] = [f"# {title}\n"]
    abstract = root.find(".//abstract")
    if abstract is not None:
        out.append("## Abstract\n")
        for p in abstract.iter("p"):
            if _text(p):
                out.append(_text(p) + "\n")

    body = root.find(".//body")
    if body is not None:
        for p in body.findall("p"):  # paragraphs with no section
            if _text(p):
                out.append(_text(p) + "\n")
        for sec in body.findall(".//sec"):
            paras = [_text(p) for p in sec.findall("p") if _text(p)]
            if not paras:
                continue
            heading = _text(sec.find("title"))
            if heading:
                out.append(f"## {heading}\n")
            out.extend(para + "\n" for para in paras)

    return meta, "\n".join(out)


def from_pdf(path: str) -> tuple[dict, str]:
    doc = fitz.open(path)
    out: list[str] = []
    for page_no, page in enumerate(doc, start=1):
        text = re.sub(r"[ \t]+", " ", page.get_text()).strip()
        if text:
            out.append(f"## Page {page_no}\n")
            out.append(text + "\n")
    doc.close()

    meta = {"title": "The Resistance Training Dose-Response (Pelland et al., 2024)",
            "authors": "Pelland, Remmert, Robinson, Hinson, Zourdos", "year": 2024,
            "source": "https://sportrxiv.org/index.php/server/preprint/view/460",
            "pmcid": "", "doi": ""}
    return meta, "\n".join(out)


def main() -> None:
    os.makedirs(TEXT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(PAPERS_DIR, "*.xml")) +
                   glob.glob(os.path.join(PAPERS_DIR, "*.pdf")))
    if not paths:
        raise SystemExit(f"No papers found in {PAPERS_DIR}")

    # license + pillar tags written by fetch_corpus.py, keyed by source filename.
    sidecar = json.load(open(CORPUS_INDEX, encoding="utf-8")) if os.path.exists(CORPUS_INDEX) else {}

    metadata: dict[str, dict] = {}
    for path in paths:
        meta, text = from_pdf(path) if path.endswith(".pdf") else from_jats(path)
        tags = sidecar.get(os.path.basename(path), {})
        meta["license"] = tags.get("license", "")
        meta["pillar"] = tags.get("pillar")
        out_name = os.path.splitext(os.path.basename(path))[0] + ".txt"
        with open(os.path.join(TEXT_DIR, out_name), "w", encoding="utf-8") as f:
            f.write(text)
        metadata[out_name] = meta
        print(f"  {os.path.basename(path):48} -> data/text/{out_name}")

    with open(os.path.join(TEXT_DIR, "_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(paths)} text files + _metadata.json")


if __name__ == "__main__":
    main()
