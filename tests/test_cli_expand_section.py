from __future__ import annotations

import json

from click.testing import CliRunner

from memsearch import cli as cli_module
from memsearch import store as store_module
from memsearch.chunker import chunk_markdown
from memsearch.cli import _extract_section, cli
from memsearch.config import MemSearchConfig

DAILY_LOG = (
    "## Session 10:00\n"
    "\n"
    "### 10:01\n"
    "<!-- session:AAA turn:t1 transcript:/tmp/a.jsonl -->\n"
    "- Fixed the login redirect bug.\n"
    "\n"
    "### 10:05\n"
    "<!-- session:AAA turn:t2 transcript:/tmp/a.jsonl -->\n"
    "- Added rate limiting to the API.\n"
)


def test_extract_section_starts_at_chunk_heading() -> None:
    lines = DAILY_LOG.splitlines()

    content, start, end = _extract_section(lines, 7, 3)

    assert (start, end) == (7, 9)
    assert content.startswith("### 10:05")
    assert "turn:t1" not in content


def test_extract_section_mid_section_chunk_walks_back_to_heading() -> None:
    lines = ["## A", "", "first paragraph", "", "second paragraph", "## B", "other"]

    content, start, end = _extract_section(lines, 5, 2)

    assert (start, end) == (1, 5)
    assert content.startswith("## A")


def test_expand_returns_the_chunks_own_section_and_anchor(monkeypatch, tmp_path) -> None:
    source = tmp_path / "2026-09-24.md"
    source.write_text(DAILY_LOG, encoding="utf-8")
    chunk = next(c for c in chunk_markdown(DAILY_LOG, str(source)) if c.heading == "10:05")

    class FakeStore:
        def __init__(self, **_kwargs):
            pass

        def query(self, filter_expr: str):
            return [
                {
                    "source": str(source),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "heading": chunk.heading,
                    "heading_level": chunk.heading_level,
                }
            ]

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli_module, "resolve_config", lambda _overrides=None: MemSearchConfig())
    monkeypatch.setattr(store_module, "MilvusStore", FakeStore)

    result = CliRunner().invoke(cli, ["expand", "abc123", "--json-output"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["start_line"] == 7
    assert data["content"].startswith("### 10:05")
    assert data["anchor"]["turn"] == "t2"
