"""
rag_engine.py
Core Retrieval-Augmented Generation engine.

Responsibilities:
- Parse PDFs into page-aware text
- Chunk text with overlap (so citations don't get cut mid-sentence)
- Embed chunks locally with Sentence Transformers (no API needed, no rate limits)
- Build / query a FAISS vector index
- Return retrieved chunks with source metadata (filename, page number, snippet)
  so the UI can highlight exactly where an answer came from
"""

import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

# This project only needs the PyTorch backend for embeddings. Some environments
# also have TensorFlow installed for unrelated work, which makes `transformers`
# (a dependency of sentence-transformers) auto-detect and try to load it —
# producing noisy oneDNN/Keras warnings that have nothing to do with this app.
# Telling it up front to skip TF avoids that whole chain, and is safe because
# we never use TF here.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=FutureWarning)

import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, free, runs locally
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap so sentences at boundaries aren't lost
INDEX_DIR = Path(__file__).parent / "data" / "indexes"


@dataclass
class Chunk:
    text: str
    filename: str
    page: int
    chunk_id: int


@dataclass
class RetrievedChunk:
    text: str
    filename: str
    page: int
    score: float


class EmbeddingModel:
    """Lazy singleton wrapper so the model is only loaded once per process."""
    _instance = None

    @classmethod
    def get(cls) -> SentenceTransformer:
        if cls._instance is None:
            cls._instance = SentenceTransformer(EMBED_MODEL_NAME)
        return cls._instance


def extract_pages(pdf_path: str) -> List[str]:
    """Return a list of raw text strings, one per page."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple sliding-window chunker with overlap. Splits on whitespace boundaries
    where possible to avoid cutting words in half."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # try to break on a space near the boundary
            space_idx = text.rfind(" ", start, end)
            if space_idx > start:
                end = space_idx
        chunks.append(text[start:end].strip())
        start = end - overlap if end - overlap > start else end
    return [c for c in chunks if c]


def build_chunks_from_pdf(pdf_path: str, filename: str) -> List[Chunk]:
    pages = extract_pages(pdf_path)
    chunks = []
    chunk_id = 0
    for page_num, page_text in enumerate(pages, start=1):
        for piece in chunk_text(page_text):
            chunks.append(Chunk(text=piece, filename=filename, page=page_num, chunk_id=chunk_id))
            chunk_id += 1
    return chunks, len(pages)


class VectorStore:
    """FAISS-backed vector store for one user session. Holds chunks from
    potentially multiple uploaded PDFs so questions can search across all of them."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.index_path = INDEX_DIR / f"{session_id}.faiss"
        self.meta_path = INDEX_DIR / f"{session_id}.json"
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        model = EmbeddingModel.get()
        if hasattr(model, "get_embedding_dimension"):
            self.dim = model.get_embedding_dimension()
        else:
            # older sentence-transformers versions don't have the renamed method
            self.dim = model.get_sentence_embedding_dimension()
        self.chunks: List[Chunk] = []
        self.index = faiss.IndexFlatIP(self.dim)  # cosine similarity via normalized vectors

        self._load_if_exists()

    def _load_if_exists(self):
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.chunks = [Chunk(**d) for d in raw]

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        # JSON, not pickle: pickle identifies classes by module+qualname and
        # requires the *exact same class object* at load time. That's fragile
        # here because Streamlit re-executes the whole script (redefining
        # Chunk fresh) on every rerun in the single-file build — pickling a
        # Chunk from one rerun and loading it in the next raises
        # "it's not the same object" even though the class is identical.
        # JSON has no such notion of class identity, so this sidesteps the
        # issue entirely regardless of how the app is structured.
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f)

    def _add_chunks(self, new_chunks: List[Chunk]):
        """Shared embed-and-index logic used by both add_pdf and add_document."""
        if not new_chunks:
            return
        texts = [c.text for c in new_chunks]
        embeddings = EmbeddingModel.get().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        self._save()

    def add_pdf(self, pdf_path: str, filename: str):
        new_chunks, num_pages = build_chunks_from_pdf(pdf_path, filename)
        self._add_chunks(new_chunks)
        return len(new_chunks), num_pages

    def add_document(self, file_path: str, filename: str):
        """Generic ingestion for any supported file type (PDF or otherwise).
        Returns (num_chunks, num_pages) just like add_pdf, so callers don't need
        to branch on file type.

        Raises:
            document_loaders.DocumentLoadError: unsupported extension
            LLMError: an image needs vision analysis but no provider is configured
        """
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return self.add_pdf(file_path, filename)

        # local import avoids a circular import (document_loaders -> llm_client,
        # nothing imports rag_engine, so this is safe to import lazily here)
        from document_loaders import load_document
        pages = load_document(file_path, filename)

        chunks = []
        chunk_id = 0
        for page_num, page_text in enumerate(pages, start=1):
            for piece in chunk_text(page_text):
                chunks.append(Chunk(text=piece, filename=filename, page=page_num, chunk_id=chunk_id))
                chunk_id += 1

        self._add_chunks(chunks)
        return len(chunks), len(pages)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if self.index.ntotal == 0:
            return []
        query_vec = EmbeddingModel.get().encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            c = self.chunks[idx]
            results.append(RetrievedChunk(text=c.text, filename=c.filename, page=c.page, score=float(score)))
        return results

    def has_documents(self) -> bool:
        return self.index.ntotal > 0

    def list_filenames(self) -> List[str]:
        return sorted(set(c.filename for c in self.chunks))

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []
        for p in (self.index_path, self.meta_path):
            if p.exists():
                p.unlink()