"""Per-turn citation numbering — stable [n] across a turn. Pure, no network."""

from app.citations import format_chunks_with_citations


def _chunk(url, title="Some Paper", content="…", year=2024):
    return {"source": url, "title": title, "content": content, "year": year}


def test_same_source_reuses_one_number():
    reg = []
    text = format_chunks_with_citations(
        [_chunk("https://a", content="first"), _chunk("https://a", content="second")],
        reg,
    )
    assert len(reg) == 1
    assert reg[0]["n"] == 1
    assert text.count("[1]") == 2      # both passages cite the same source


def test_distinct_sources_get_incrementing_numbers():
    reg = []
    text = format_chunks_with_citations(
        [_chunk("https://a", title="A"), _chunk("https://b", title="B")],
        reg,
    )
    assert [r["n"] for r in reg] == [1, 2]
    assert {r["url"] for r in reg} == {"https://a", "https://b"}
    assert "[1]" in text and "[2]" in text


def test_existing_registry_is_respected_across_calls():
    reg = [{"n": 1, "title": "A", "url": "https://a"}]
    # A second knowledge_search call in the same turn: known source keeps its
    # number, a new source continues from where the registry left off.
    format_chunks_with_citations([_chunk("https://a"), _chunk("https://b")], reg)
    by_url = {r["url"]: r["n"] for r in reg}
    assert by_url == {"https://a": 1, "https://b": 2}
