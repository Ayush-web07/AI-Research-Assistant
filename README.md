# 📚 AI Research Assistant

A RAG-powered research assistant that lets you upload multiple PDFs, ask questions
across all of them, and get grounded, cited answers — with source paragraphs
highlighted so you can verify every claim.

Built for the intermediate-GenAI-portfolio use case: multi-document semantic
search, citation grounding, persistent chat history, and PDF export — all on
100% free APIs, no credit card required.

## Features

- 📤 Upload multiple PDFs, indexed for cross-document search
- 💬 Ask natural-language questions, get grounded answers
- 🔍 Semantic search via FAISS + local Sentence Transformer embeddings (no rate limits, no cost)
- 📎 Every answer shows the exact source paragraphs it was built from (filename, page, similarity score)
- 📝 One-click document summarization
- 💾 Chat history persisted per session in SQLite
- 📄 Export the full Q&A conversation to a formatted PDF
- ⚡ Fast generation via Groq's free-tier Llama models (Hugging Face Inference API and fully-offline Ollama as fallbacks)

## Architecture

```
PDF Upload → pdfplumber (text extraction, page-aware)
           → chunk_text() (800 chars, 150 overlap, sentence-safe)
           → Sentence Transformers "all-MiniLM-L6-v2" (local embeddings)
           → FAISS IndexFlatIP (cosine similarity search)

Question   → embed query → FAISS top-k retrieval
           → build cited context block
           → Groq / HF LLM → grounded answer with [Source N] citations

Everything → SQLite (chat_history, documents tables)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Choose an LLM provider (no credit card needed for any of these)

Pick **one**:

- **Groq (recommended — much faster)**: sign up at https://console.groq.com/keys
- **Hugging Face (fallback)**: sign up at https://huggingface.co/settings/tokens
- **Ollama (fully offline, no API key at all)**: install from https://ollama.com/download,
  then run `ollama pull llama3.1` and start it with `ollama serve` (or just open the app)

The app auto-detects which one is available (in that order) — no config needed if
you only set up one. To force a specific provider, set `LLM_PROVIDER=groq|hf|ollama`.

### 3. Set your API key (skip this step if you're only using Ollama)

```bash
cp .env.example .env
# edit .env and paste your key
export GROQ_API_KEY=gsk_your_key_here      # or
export HF_API_TOKEN=hf_your_token_here
```

(On Windows PowerShell: `$env:GROQ_API_KEY="gsk_your_key_here"`)

If you're using Ollama, no key is needed — just make sure `ollama serve` is running
before you launch the app.

### 4. Run the app

```bash
streamlit run app.py
```

The first run will download the embedding model (~90MB) — this requires
internet access one time, then it runs fully offline for embeddings.

## Project structure

```
ai-research-assistant/
├── app.py              # Streamlit UI — upload, chat, history, export
├── rag_engine.py        # PDF parsing, chunking, embeddings, FAISS index
├── llm_client.py         # Groq / Hugging Face API wrapper + prompting
├── database.py            # SQLite persistence (documents, chat history)
├── export_utils.py         # Chat → PDF export (reportlab)
├── requirements.txt
├── .env.example
└── data/                    # SQLite DB + FAISS indexes (created at runtime)
```

## Notes for recruiters / interview talking points

- **Chunking strategy**: sliding-window with overlap, breaking on whitespace
  boundaries so citations aren't cut mid-sentence — a common failure mode in
  naive RAG implementations.
- **Grounding**: the system prompt forces the model to cite `[Source N]` and
  explicitly say when the answer isn't in the retrieved context, reducing
  hallucination.
- **Cost/latency tradeoff**: embeddings run locally (free, no rate limit,
  no latency from network calls per chunk); only the final generation step
  calls an external API, minimizing API usage and cost.
- **Extensibility**: swapping the embedding model, chunk size, or LLM
  provider is a one-line change in `rag_engine.py` / `llm_client.py`.

## Possible extensions

- Streaming responses (Groq supports SSE streaming)
- Re-ranking retrieved chunks with a cross-encoder before generation
- Multi-turn conversational memory (currently each question is independent)
- Support for other file types (docx, txt, web pages)
