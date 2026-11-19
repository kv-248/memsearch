"""Search scoping must treat source prefixes as literal strings.

Milvus ``like`` patterns treat ``_`` and ``%`` as wildcards. A prefix such as
``my_app`` must not also return ``myXapp``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memsearch.core import MemSearch
from memsearch.store import MilvusStore, _prefix_upper_bound, source_prefix_filter


class _SameVectorEmbedder:
    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 4

    @property
    def batch_size(self) -> int:
        return 32

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _write(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True)
    path = directory / "notes.md"
    path.write_text(f"# Notes\n\n{body}\n", encoding="utf-8")
    return path


def test_prefix_upper_bound_is_exact_for_literal_prefixes() -> None:
    prefixes = ["", "my_app", "100%done", "ab\U0010ffff", "\U0010ffff", "a\U0010ffff\U0010ffff"]
    samples = ["", "x", "myXapp", "my_app", "my_app/notes.md", "100Xdone", "100%done/x", "ac"]
    for prefix in prefixes:
        upper = _prefix_upper_bound(prefix)
        candidates = [*samples, prefix, prefix + "tail", prefix[:-1]]
        for sample in candidates:
            starts = sample.startswith(prefix)
            in_range = sample >= prefix and (upper is None or sample < upper)
            assert starts is in_range, (prefix, upper, sample)


def test_source_prefix_filter_escapes_backslashes_and_quotes(tmp_path: Path) -> None:
    prefix = 'C:\\data\\my_app\\say"hi'
    match = prefix + "\\notes.md"
    decoy = 'C:\\data\\myXapp\\say"hi\\notes.md'
    store = MilvusStore(uri=str(tmp_path / "milvus.db"), dimension=4, _create_if_missing=True)
    try:
        store.upsert(
            [
                {
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                    "content": "widgets in the windows tree",
                    "source": source,
                    "heading": "Notes",
                    "chunk_hash": f"h{index}",
                    "heading_level": 1,
                    "start_line": 1,
                    "end_line": 3,
                }
                for index, source in enumerate((match, decoy))
            ]
        )
        hits = store.search(
            [1.0, 0.0, 0.0, 0.0],
            query_text="widgets",
            top_k=10,
            filter_expr=source_prefix_filter(prefix),
        )
    finally:
        store.close()

    assert {hit["source"] for hit in hits} == {match}


@pytest.mark.asyncio
async def test_search_source_prefix_does_not_treat_like_wildcards_as_special(tmp_path: Path) -> None:
    app = _write(tmp_path / "my_app", "alpha widgets live here")
    extra = _write(tmp_path / "my_app-extra", "epsilon widgets live next door")
    underscore_decoy = _write(tmp_path / "myXapp", "beta widgets should stay out")
    percent = _write(tmp_path / "100%done", "gamma widgets are complete")
    percent_decoy = _write(tmp_path / "100Xdone", "delta widgets are a different directory")

    ms = MemSearch.__new__(MemSearch)
    ms._paths = [str(tmp_path)]
    ms._max_chunk_size = 1500
    ms._overlap_lines = 2
    ms._ignore_files = []
    ms._exclude = []
    ms._embedder = _SameVectorEmbedder()
    ms._reranker_model = ""
    ms._store = MilvusStore(uri=str(tmp_path / "milvus.db"), dimension=4, _create_if_missing=True)
    try:
        await ms.index()

        underscore_hits = await ms.search("widgets", top_k=10, source_prefix=tmp_path / "my_app")
        underscore_sources = {hit["source"] for hit in underscore_hits}
        assert str(app.resolve()) in underscore_sources
        assert str(extra.resolve()) in underscore_sources
        assert str(underscore_decoy.resolve()) not in underscore_sources

        percent_hits = await ms.search("widgets", top_k=10, source_prefix=tmp_path / "100%done")
        percent_sources = {hit["source"] for hit in percent_hits}
        assert str(percent.resolve()) in percent_sources
        assert str(percent_decoy.resolve()) not in percent_sources
    finally:
        ms.close()
