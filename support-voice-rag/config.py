"""All settings in one place."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # silence Chroma telemetry

DATA_DIR = ROOT / "data"
STORAGE_DIR = ROOT / "storage"
DB_DIR = STORAGE_DIR / "chroma"
SQLITE_PATH = STORAGE_DIR / "memory.db"
TICKETS_PATH = STORAGE_DIR / "handoffs.jsonl"

# Multilingual embeddings: Arabic and English questions both match the same PDFs
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
TTS_MODEL = os.getenv("TTS_MODEL", "canopylabs/orpheus-v1-english")
TTS_VOICE = os.getenv("TTS_VOICE", "autumn")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
HISTORY_TURNS = 6      # past exchanges sent to the model
TTS_CHUNK_CHARS = 190  # TTS requests are kept short and joined afterwards
