"""Per-turn citation numbering — pure, so it's unit-testable on its own.

A single agent turn may call `knowledge_search` several times. We want [n] to
mean the same paper across the whole turn, so we keep a registry (a list of
{n, title, url}) and hand each distinct source a stable global number the first
time we see it.
"""

from __future__ import annotations


def format_chunks_with_citations(chunks: list[dict], registry: list[dict]) -> str:
    """Render retrieved chunks with stable [n] citation markers.

    `registry` is mutated in place: each new source url is appended as
    {n, title, url} and reused if it appears again later in the same turn.
    """
    url_to_n = {c["url"]: c["n"] for c in registry}

    lines = []
    for ch in chunks:
        url = ch["source"]
        n = url_to_n.get(url)
        if n is None:
            n = len(registry) + 1
            registry.append({"n": n, "title": ch["title"], "url": url})
            url_to_n[url] = n
        lines.append(f"[{n}] {ch['content']}\n(Source: {ch['title']}, {ch['year']})")
    return "\n\n".join(lines)
