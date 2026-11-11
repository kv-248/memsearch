from __future__ import annotations

import json
from pathlib import Path

import pytest

from memsearch import transcript as tr


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_claude_format_extracts_tool_command_and_output(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "c.jsonl",
        [
            {"type": "user", "uuid": "u1", "message": {"content": "run the tests"}},
            {
                "type": "assistant",
                "uuid": "a1",
                "message": {
                    "content": [
                        {"type": "text", "text": "Running them"},
                        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest -x --ff tests/"}},
                    ]
                },
            },
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "5 passed"}]},
            },
        ],
    )
    assert tr.detect_format(tr._load_jsonl(p)) == "claude"
    turns = tr.parse_transcript(p)
    assert [t.role for t in turns] == ["user", "assistant"]
    tc = turns[1].tools[0]
    assert tc.command == "pytest -x --ff tests/"  # full command, not truncated
    assert "5 passed" in tc.output


def test_codex_rollout_extracts_function_call(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "r.jsonl",
        [
            {"type": "event_msg", "payload": {"type": "user_message", "message": "deploy to staging"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "c1",
                    "arguments": json.dumps({"command": ["bash", "-lc", "make build-fast && kubectl apply -f k8s/"]}),
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "function_call_output", "call_id": "c1", "output": "deployed"},
            },
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "done"}},
        ],
    )
    assert tr.detect_format(tr._load_jsonl(p)) == "codex"
    turns = tr.parse_transcript(p)
    tools = [tc for t in turns for tc in t.tools]
    assert "make build-fast && kubectl apply -f k8s/" in tools[0].command  # exact command recovered
    assert tools[0].output == "deployed"


def test_openclaw_format_extracts_tool_call(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "o.jsonl",
        [
            {"type": "message", "id": "m1", "message": {"role": "user", "content": [{"type": "text", "text": "lint"}]}},
            {
                "type": "message",
                "id": "m2",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "ok"},
                        {"type": "toolCall", "name": "bash", "input": {"command": "ruff check ."}},
                        {"type": "toolResult", "text": "clean"},
                    ],
                },
            },
        ],
    )
    assert tr.detect_format(tr._load_jsonl(p)) == "openclaw"
    turns = tr.parse_transcript(p)
    tc = turns[1].tools[0]
    assert tc.command == "ruff check ."
    assert tc.output == "clean"


def test_unknown_format_raises(tmp_path: Path) -> None:
    p = _write(tmp_path / "x.jsonl", [{"foo": "bar"}, {"baz": 1}])
    with pytest.raises(tr.UnknownTranscriptFormat):
        tr.parse_transcript(p)


def test_format_turns_includes_command(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "c.jsonl",
        [
            {"type": "user", "uuid": "u1", "message": {"content": "go"}},
            {
                "type": "assistant",
                "uuid": "a1",
                "message": {
                    "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "uv run pytest"}}]
                },
            },
        ],
    )
    rendered = tr.format_turns(tr.parse_transcript(p))
    assert "uv run pytest" in rendered


def test_select_turns_by_id(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "c.jsonl",
        [
            {"type": "user", "uuid": "aaaa1111", "message": {"content": "one"}},
            {"type": "user", "uuid": "bbbb2222", "message": {"content": "two"}},
            {"type": "user", "uuid": "cccc3333", "message": {"content": "three"}},
        ],
    )
    turns = tr.parse_transcript(p)
    sel = tr.select_turns(turns, "bbbb2222", context=0)
    assert [t.text for t in sel] == ["two"]


def test_select_turns_prefix_does_not_stop_at_a_shared_eight_char_stem() -> None:
    """A later id that only shares the first eight characters must not lose to an earlier turn.

    ``--turn`` is a prefix of the stored id. Comparing the query against only
    ``uuid[:8]`` in the other direction selects the first turn whose id begins
    with those characters, even when the rest of the query names a later turn.
    """
    first = "aaaaaaaa-1111-1111-1111-111111111111"
    second = "aaaaaaaa-2222-2222-2222-222222222222"
    third = "bbbbbbbb-3333-3333-3333-333333333333"
    turns = [
        tr.Turn(role="user", uuid=first, text="first"),
        tr.Turn(role="user", uuid=second, text="second"),
        tr.Turn(role="user", uuid="m1", text="one"),
        tr.Turn(role="user", uuid="m10", text="ten"),
        tr.Turn(role="user", uuid=third, text="third"),
    ]

    assert [t.text for t in tr.select_turns(turns, second, context=0)] == ["second"]
    assert [t.text for t in tr.select_turns(turns, "aaaaaaaa-2222", context=0)] == ["second"]
    assert [t.text for t in tr.select_turns(turns, "m10", context=0)] == ["ten"]
    # A short prefix is still the first stored id that begins with it.
    assert [t.text for t in tr.select_turns(turns, "aaaaaaaa", context=0)] == ["first"]
    assert [t.text for t in tr.select_turns(turns, "bbbbbbbb", context=1)] == ["ten", "third"]


def test_tool_output_is_clipped(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "c.jsonl",
        [
            {"type": "user", "uuid": "u1", "message": {"content": "go"}},
            {
                "type": "assistant",
                "uuid": "a1",
                "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "x"}}]},
            },
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Z" * 5000}]},
            },
        ],
    )
    turns = tr.parse_transcript(p)
    assert len(turns[1].tools[0].output) <= tr.MAX_OUTPUT_CHARS + 20  # clipped, not the full 5000


def test_render_is_capped(tmp_path: Path) -> None:
    rows = [{"type": "user", "uuid": f"u{i}", "message": {"content": "x" * 60}} for i in range(2000)]
    out = tr.format_turns(tr.parse_transcript(_write(tmp_path / "big.jsonl", rows)))
    assert len(out) <= tr.MAX_RENDER_CHARS + 80
    assert "truncated" in out
