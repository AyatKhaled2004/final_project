"""End-to-end UI test with fake models (no network, no API keys)."""
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_missing_key_shows_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert at.error and "GROQ_API_KEY" in at.error[0].value


def test_typed_question_gets_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    import config, langchain_groq
    from core import store, voice
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "m.db")
    monkeypatch.setattr(config, "DB_DIR", tmp_path / "chroma")
    monkeypatch.setattr(store, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(langchain_groq, "ChatGroq",
                        lambda **kw: FakeListChatModel(responses=["knowledge", "Fourteen days."]))
    monkeypatch.setattr(voice, "speak", lambda t: None)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.chat_input[0].set_value("How long to return an item?").run()
    assert not at.exception
    texts = [m.markdown[0].value for m in at.chat_message]
    assert any("Fourteen days." in t for t in texts)
