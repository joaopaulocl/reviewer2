"""call_claude request shape and response handling, against a fake client."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reviewer2 import core


def _msg(stop_reason="end_turn", texts=("OK",)):
    content = [SimpleNamespace(type="thinking", thinking="")] + [
        SimpleNamespace(type="text", text=t) for t in texts]
    usage = SimpleNamespace(input_tokens=10, output_tokens=5,
                            cache_creation_input_tokens=100, cache_read_input_tokens=0)
    return SimpleNamespace(content=content, stop_reason=stop_reason, stop_details=None,
                           usage=usage, model="claude-opus-5-5")


class _FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        fake = self

        class _Stream:
            def __init__(self, reply):
                self.reply = reply

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get_final_message(self):
                return self.reply

        def stream(**kwargs):
            fake.calls.append(kwargs)
            return _Stream(fake.replies.pop(0))

        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=stream))
        self.files = SimpleNamespace(upload=lambda **k: SimpleNamespace(id="file_123"))


@pytest.fixture
def fake(monkeypatch, tmp_path):
    def install(*replies):
        client = _FakeClient(replies)
        monkeypatch.setattr(core, "_get_client", lambda timeout: client)
        monkeypatch.setattr(core, "_FILE_CACHE", {})
        monkeypatch.setattr(core.time, "sleep", lambda s: None)
        monkeypatch.delenv("REVIEWER2_MODEL_OVERRIDE", raising=False)
        return client
    return install


def test_request_shape_with_pdf(fake, tmp_path):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    client = fake(_msg(texts=("Hello ", "world")))
    out = core.call_claude("Review it", str(pdf), model_type="pro_3_1", temperature=0.4,
                           thinking_level="high", system_instruction="Be harsh.", step="s")
    assert out == "Hello world"
    req = client.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"] == {"effort": "high"}
    assert req["system"] == "Be harsh."
    assert "temperature" not in req and "thinking" not in req
    assert req["fallbacks"] == "default"
    doc, text = req["messages"][0]["content"]
    assert doc["source"] == {"type": "file", "file_id": "file_123"}
    assert doc["cache_control"]["ttl"] == "1h"
    assert text == {"type": "text", "text": "Review it"}
    assert core.USAGE_LOG[-1]["cache_write_tokens"] == 100


@pytest.mark.parametrize("level,budget,effort", [
    (None, 0, "low"), (None, 1000, "low"), (None, -1, "medium"), (None, None, "medium"),
    ("low", None, "low"),
])
def test_effort_mapping(level, budget, effort):
    assert core._effort_for(level, budget) == effort


def test_search_tool_and_small_cap_raised(fake):
    client = fake(_msg())
    core.call_claude("Find it", use_search=True, max_output_tokens=2000)
    req = client.calls[0]
    assert req["tools"] == [{"type": "web_search_20260209", "name": "web_search"}]
    assert req["max_tokens"] >= 16000


def test_pause_turn_resumes(fake):
    paused = _msg(stop_reason="pause_turn", texts=())
    client = fake(paused, _msg(texts=("done",)))
    assert core.call_claude("x", use_code_exec=True) == "done"
    second = client.calls[1]["messages"]
    assert second[1] == {"role": "assistant", "content": paused.content}


def test_refusal_is_fatal(fake):
    fake(_msg(stop_reason="refusal", texts=()))
    with pytest.raises(RuntimeError, match="FATAL"):
        core.call_claude("x")


def test_empty_text_retries(fake):
    client = fake(_msg(texts=("",)), _msg(texts=("second",)))
    assert core.call_claude("x") == "second"
    assert len(client.calls) == 2
