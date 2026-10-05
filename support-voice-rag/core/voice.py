"""Speech: Groq Whisper for speech-to-text, Groq TTS for text-to-speech."""
import io
import logging
import re
import wave

import config

log = logging.getLogger("voice")
_client = None


def _groq():
    global _client
    if _client is None:
        from groq import Groq

        _client = Groq()
    return _client


def transcribe(audio: bytes, language: str | None = None, filename: str = "recording.wav") -> str:
    kwargs = {"model": config.STT_MODEL, "file": (filename, audio)}
    if language:
        kwargs["language"] = language
    return (_groq().audio.transcriptions.create(**kwargs).text or "").strip()


def clean_for_speech(text: str) -> str:
    text = re.sub(r"\(Source:.*?\)", "", text)
    text = re.sub(r"[*_`#>]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str, limit: int = config.TTS_CHUNK_CHARS) -> list[str]:
    sentences = re.split(r"(?<=[.!?\u061f\n])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        while len(s) > limit:  # very long sentence: cut on a space
            cut = s.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                chunks.append(current)
                current = ""
            chunks.append(s[:cut].strip())
            s = s[cut:].strip()
        if len(current) + len(s) + 1 <= limit:
            current = f"{current} {s}".strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def join_wavs(parts: list[bytes]) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        for i, part in enumerate(parts):
            with wave.open(io.BytesIO(part), "rb") as reader:
                if i == 0:
                    writer.setparams(reader.getparams())
                writer.writeframes(reader.readframes(reader.getnframes()))
    return out.getvalue()


def speak(text: str) -> bytes | None:
    """Return WAV bytes, or None if TTS is unavailable (the chat keeps working)."""
    text = clean_for_speech(text)
    if not text:
        return None
    try:
        parts = []
        for chunk in split_text(text):
            resp = _groq().audio.speech.create(
                model=config.TTS_MODEL, voice=config.TTS_VOICE, input=chunk, response_format="wav"
            )
            parts.append(resp.read())
        return join_wavs(parts)
    except Exception as exc:
        log.warning("TTS failed: %s", exc)
        return None
