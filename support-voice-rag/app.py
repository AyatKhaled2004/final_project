"""Streamlit UI: one chat box - type, or tap the microphone inside it."""
import logging
import os
import uuid

import streamlit as st
from langchain_groq import ChatGroq

import config
from core import memory, voice
from core.graph import build_graph
from core.handoff import create_ticket
from core.store import KnowledgeBase

st.set_page_config(page_title="Support Assistant", page_icon="🎧", layout="centered")

# let the browser pick text direction per paragraph (Arabic and English mix fine)
st.markdown(
    "<style>.stChatMessage p, .stChatMessage li {unicode-bidi: plaintext; text-align: start;}</style>",
    unsafe_allow_html=True,
)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Copy .env.example to .env and add your key, then restart.")
    st.stop()


@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_backend():
    kb = KnowledgeBase()
    status = kb.sync()
    llm = ChatGroq(model=config.LLM_MODEL, temperature=0)
    return kb, build_graph(llm, kb, on_handoff=create_ticket), status


kb, graph, kb_status = load_backend()

ss = st.session_state
ss.setdefault("session_id", uuid.uuid4().hex)
ss.setdefault("messages", [])
ss.setdefault("autoplay", False)

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    speak_on = st.toggle("Read answers aloud", value=memory.get_pref("speak", "0") == "1")
    lang_label = st.selectbox(
        "Voice language", ["Auto", "English", "Arabic"],
        index=["Auto", "English", "Arabic"].index(memory.get_pref("stt_lang", "Auto")),
    )
    memory.set_pref("speak", "1" if speak_on else "0")
    memory.set_pref("stt_lang", lang_label)
    stt_lang = {"Auto": None, "English": "en", "Arabic": "ar"}[lang_label]

    st.caption(kb_status)
    if st.button("Re-index PDFs"):
        with st.spinner("Indexing..."):
            st.success(kb.sync(force=True))
    if st.button("New conversation"):
        ss.messages = []
        ss.session_id = uuid.uuid4().hex
        st.rerun()
    st.caption("Put your PDFs in the data/ folder, then click Re-index (it also re-indexes on restart when files change).")

st.title("Support Assistant")
st.caption("Type a message, or tap the microphone in the box and speak.")


def handle(question: str, via_voice: bool) -> None:
    history = [(m["role"], m["content"]) for m in ss.messages][-config.HISTORY_TURNS * 2:]
    ss.messages.append({"role": "user", "content": question, "voice": via_voice})
    memory.add_message(ss.session_id, "user", question)
    try:
        result = graph.invoke({"question": question, "history": history, "session_id": ss.session_id})
    except Exception as exc:
        logging.getLogger("app").exception("Question failed")  # full traceback in the terminal
        detail = f"{exc.__class__.__name__}: {exc}"[:400]
        ss.messages.append({"role": "assistant", "content": f"Sorry, something went wrong.\n\n`{detail}`"})
        return
    answer = result["answer"]
    audio = voice.speak(answer) if speak_on else None
    ss.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": result.get("sources", []),
            "handoff": result.get("handoff", False),
            "audio": audio,
        }
    )
    memory.add_message(ss.session_id, "assistant", answer)
    ss.autoplay = audio is not None


# ---------------- chat history ----------------
last = len(ss.messages) - 1
for i, m in enumerate(ss.messages):
    with st.chat_message(m["role"]):
        st.markdown(("🎙️ " if m.get("voice") else "") + m["content"])
        if m.get("sources"):
            st.caption("Sources: " + " · ".join(m["sources"]))
        if m.get("handoff"):
            st.info("A ticket was created for a human agent.")
        if m.get("audio"):
            st.audio(m["audio"], format="audio/wav", autoplay=(i == last and ss.autoplay))
ss.autoplay = False

# ---------------- input: one chat box with a built-in microphone ----------------
def read_input(value):
    """chat_input returns None, a str, or an object with .text and .audio."""
    if value is None:
        return "", None
    if isinstance(value, str):
        return value, None
    return (value.text or ""), getattr(value, "audio", None)


typed, recording = read_input(
    st.chat_input("Type a message or tap the mic...", accept_audio=True, audio_sample_rate=16000)
)

question, via_voice, failed = None, False, False
if recording is not None:
    with st.spinner("Transcribing..."):
        try:
            question = voice.transcribe(recording.getvalue(), stt_lang)
        except Exception as exc:
            failed = True
            logging.getLogger("app").exception("Transcription failed")
            st.error(f"Could not transcribe the recording: {exc.__class__.__name__}: {str(exc)[:300]}")
    via_voice = True
    if not question and not failed:
        st.warning("I could not hear anything in that recording. Please try again.")
elif typed.strip():
    question = typed.strip()

if question:
    with st.spinner("Thinking..."):
        handle(question, via_voice)
    st.rerun()