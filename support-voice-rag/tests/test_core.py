import io
import wave
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import config
from core import memory, voice
from core.graph import build_graph, parse_route
from core.store import KnowledgeBase

PDF = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.pdf"


@pytest.fixture()
def kb(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "kb.pdf").write_bytes(PDF.read_bytes())
    base = KnowledgeBase(data, tmp_path / "db", embeddings=DeterministicFakeEmbedding(size=32))
    base.sync()
    return base


def test_sync_indexes_then_skips_when_unchanged(kb):
    assert "up to date" in kb.sync()
    assert "Indexed" in kb.sync(force=True)


def test_search_returns_pdf_chunks(kb):
    docs = kb.search("refund policy")
    assert docs and docs[0].metadata["page"] >= 0


def test_parse_route():
    assert parse_route("human") == "human"
    assert parse_route("Chitchat.") == "chitchat"
    assert parse_route("knowledge") == "knowledge"
    assert parse_route("???") == "knowledge"


def test_knowledge_route(kb):
    llm = FakeListChatModel(responses=["knowledge", "You can return items within 14 days."])
    out = build_graph(llm, kb).invoke({"question": "How long do I have to return an item?", "history": []})
    assert out["route"] == "knowledge" and "14 days" in out["answer"]
    assert out["sources"] and not out["handoff"]


def test_followup_is_rewritten(kb):
    llm = FakeListChatModel(responses=["What is the return window?", "knowledge", "Fourteen days."])
    out = build_graph(llm, kb).invoke(
        {"question": "and how long?", "history": [("user", "Can I return an item?"), ("assistant", "Yes.")]}
    )
    assert out["standalone"] == "What is the return window?"


def test_chitchat_route(kb):
    llm = FakeListChatModel(responses=["chitchat", "Hello! How can I help?"])
    out = build_graph(llm, kb).invoke({"question": "hi", "history": []})
    assert out["route"] == "chitchat" and out["sources"] == []


def test_handoff_creates_ticket(kb):
    tickets = []
    llm = FakeListChatModel(responses=["human", "Sorry, a human agent will follow up."])
    out = build_graph(llm, kb, on_handoff=tickets.append).invoke(
        {"question": "I want a person NOW", "history": [], "session_id": "s1"}
    )
    assert out["handoff"] and tickets[0]["session_id"] == "s1"


def test_memory_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "m.db")
    memory.add_message("a", "user", "hi")
    memory.add_message("a", "assistant", "hello")
    assert memory.recent_messages("a") == [("user", "hi"), ("assistant", "hello")]
    memory.set_pref("speak", "1")
    assert memory.get_pref("speak") == "1" and memory.get_pref("nope", "x") == "x"


def _wav(frames: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def test_tts_helpers():
    long_text = ("This is a sentence. " * 30).strip()
    chunks = voice.split_text(long_text, limit=100)
    assert all(len(c) <= 100 for c in chunks) and " ".join(chunks).replace("  ", " ")
    joined = voice.join_wavs([_wav(100), _wav(50)])
    with wave.open(io.BytesIO(joined)) as w:
        assert w.getnframes() == 150
    assert voice.clean_for_speech("**Hi** (Source: PDF, confidence: High)") == "Hi"
