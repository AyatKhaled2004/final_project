"""PDF knowledge base: load, split, embed, store in Chroma, search."""
import hashlib
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)


def format_source(doc: Document) -> str:
    name = Path(doc.metadata.get("source", "document")).name
    page = doc.metadata.get("page")
    return f"{name} (p. {int(page) + 1})" if page is not None else name


class KnowledgeBase:
    def __init__(self, data_dir: Path = config.DATA_DIR, db_dir: Path = config.DB_DIR, embeddings=None):
        self.data_dir = Path(data_dir)
        self.db_dir = Path(db_dir)
        self._embeddings = embeddings
        self._db: Chroma | None = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def _pdfs(self) -> list[Path]:
        return sorted(self.data_dir.glob("*.pdf"))

    def _fingerprint(self, pdfs: list[Path]) -> str:
        h = hashlib.sha256()
        for pdf in pdfs:
            h.update(pdf.name.encode())
            h.update(pdf.read_bytes())
        return h.hexdigest()

    def _open(self) -> Chroma:
        self.db_dir.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name="knowledge_base",
            persist_directory=str(self.db_dir),
            embedding_function=self.embeddings,
        )

    def sync(self, force: bool = False) -> str:
        """Re-index only when the PDFs changed (or force=True)."""
        pdfs = self._pdfs()
        if not pdfs:
            self._db = None
            return "No PDF files found in the data folder."

        manifest = self.db_dir / "manifest.json"
        fingerprint = self._fingerprint(pdfs)
        previous = json.loads(manifest.read_text())["fingerprint"] if manifest.exists() else None

        db = self._open()
        if not force and previous == fingerprint and db.get(limit=1)["ids"]:
            self._db = db
            return f"Knowledge base is up to date ({len(pdfs)} PDF file(s))."

        docs: list[Document] = []
        for pdf in pdfs:
            docs.extend(PyPDFLoader(str(pdf)).load())
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)

        db.delete_collection()
        db = self._open()
        db.add_documents(chunks)
        manifest.write_text(json.dumps({"fingerprint": fingerprint}))
        self._db = db
        return f"Indexed {len(chunks)} chunks from {len(pdfs)} PDF file(s)."

    def search(self, query: str, k: int = config.TOP_K) -> list[Document]:
        if self._db is None:
            return []
        return self._db.similarity_search(query, k=k)
