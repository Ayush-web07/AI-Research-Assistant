"""
app.py
AI Research Assistant — Streamlit front-end.

Run with:
    streamlit run app.py

Features:
- Upload multiple PDFs
- AI Tools sidebar panel: Dashboard, Chat with PDF, Chat with Image, Quiz &
  Flashcards, Mind Map, Compare Documents, Deep Research, Translate, Voice
  Chat — one tool shown at a time in the main area, selected from the sidebar
- Grounded answers with source citations, highlighted source paragraphs
- Persistent chat history (SQLite) per browser session
- Export full Q&A conversation to PDF
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# On Windows, Python's default "Proactor" asyncio event loop logs a noisy (but
# harmless) ConnectionResetError traceback whenever a browser tab/WebSocket
# disconnects abruptly. Switching to the "Selector" event loop policy avoids
# this. Must happen before Streamlit's own event loop starts, so it's the
# very first thing this file does.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json

import streamlit as st
from dotenv import load_dotenv, set_key

import database as db
from rag_engine import VectorStore
from llm_client import generate_answer_stream, LLMError
from quiz_generator import generate_quiz, QUIZ_TYPES, DIFFICULTIES, QuizGenerationError
from mindmap_generator import generate_mindmap, tree_to_graph, render_outline, MindMapError
from doc_comparison import compare_documents, DocComparisonError
from deep_research import run_deep_research, DeepResearchError
from image_analysis import describe_page
from pdf_vision import get_page_count, PDFVisionError
from document_loaders import SUPPORTED_EXTENSIONS, DocumentLoadError
from translator import translate_text
from voice_client import transcribe_audio, synthesize_speech, TTS_VALID_VOICES, TTS_DEFAULT_MAX_TOTAL_CHARS
from export_utils import export_chat_to_pdf

# ---------------------------------------------------------------------------
# Load saved API keys from .env (created next to this file) so they persist
# across terminal sessions — no need to re-run `$env:GROQ_API_KEY=...` every time.
# ---------------------------------------------------------------------------
ENV_PATH = Path(__file__).parent / ".env"
if not ENV_PATH.exists():
    ENV_PATH.touch()
load_dotenv(ENV_PATH, override=True)

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")

# Persistent storage for uploaded PDFs, keyed by session, so features that need
# the original file (e.g. rendering a page as an image for chart analysis) can
# re-open it later — the RAG index only stores extracted text, not the file itself.
UPLOADS_DIR = Path(__file__).parent / "data" / "uploads"


def _uploaded_file_path(session_id: str, filename: str) -> Path:
    return UPLOADS_DIR / session_id / filename


# ---------------------------------------------------------------------------
# AI Tools registry — each entry drives both the sidebar nav and the main-area
# header. Only tools that are actually implemented are listed here.
# ---------------------------------------------------------------------------
TOOLS = [
    {"key": "dashboard", "label": "📊 Dashboard",
     "caption": "An overview of this session's activity across every tool."},
    {"key": "chat", "label": "💬 Chat with PDF",
     "caption": "Semantic search across all uploaded PDFs, with cited, highlighted sources."},
    {"key": "image", "label": "🖼️ Chat with Image",
     "caption": "Ask a vision model to read charts, diagrams, tables, and photos on any page."},
    {"key": "quiz", "label": "🎯 Quiz & Flashcards",
     "caption": "Generate MCQs, coding questions, interview questions, or flashcards."},
    {"key": "mindmap", "label": "🧠 Mind Map",
     "caption": "Visualize any topic — or a document's key concepts — as an interactive graph."},
    {"key": "compare", "label": "⚖️ Compare Documents",
     "caption": "Find similarities, differences, missing clauses, and risks between two documents."},
    {"key": "research", "label": "🔬 Deep Research",
     "caption": "Search plan, step-by-step reasoning, and an honest confidence score."},
    {"key": "translate", "label": "🌍 Translate",
     "caption": "Translate pasted text or an uploaded document into any language."},
    {"key": "voice", "label": "🎤 Voice Chat",
     "caption": "Ask your documents a question out loud instead of typing."},
]

# ---------------------------------------------------------------------------
# Session state / persistence setup
# ---------------------------------------------------------------------------
db.init_db()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.utcnow()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(st.session_state.session_id)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = db.get_chat_history(st.session_state.session_id)

vs: VectorStore = st.session_state.vector_store


def _render_mindmap_html(tree: dict) -> str:
    """Build a self-contained HTML snippet that renders the given mind-map tree as
    an interactive, draggable graph using vis-network (loaded from a CDN)."""
    nodes, edges = tree_to_graph(tree)

    level_colors = {
        0: {"background": "#d97706", "border": "#92400e"},  # root — amber
        1: {"background": "#2563eb", "border": "#1e3a8a"},  # branches — blue
    }
    default_color = {"background": "#16a34a", "border": "#14532d"}  # leaves — green

    vis_nodes = []
    for n in nodes:
        color = level_colors.get(n["level"], default_color)
        vis_nodes.append({
            "id": n["id"],
            "label": n["label"],
            "level": n["level"],
            "color": color,
            "font": {"color": "#ffffff", "size": 15 if n["level"] == 0 else 13},
        })

    nodes_json = json.dumps(vis_nodes).replace("</", "<\\/")
    edges_json = json.dumps(edges).replace("</", "<\\/")

    return f"""
    <div id="mindmap" style="height: 520px; border: 1px solid #444; border-radius: 10px; background: #0e1117;"></div>
    <script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
    <script>
      const nodes = new vis.DataSet({nodes_json});
      const edges = new vis.DataSet({edges_json});
      const container = document.getElementById('mindmap');
      const data = {{ nodes: nodes, edges: edges }};
      const options = {{
        layout: {{
          hierarchical: {{
            direction: 'UD',
            sortMethod: 'directed',
            nodeSpacing: 160,
            levelSeparation: 130
          }}
        }},
        nodes: {{
          shape: 'box',
          margin: 10,
          borderWidth: 2,
          shapeProperties: {{ borderRadius: 6 }}
        }},
        edges: {{
          color: {{ color: '#555555', highlight: '#999999' }},
          smooth: {{ type: 'cubicBezier', forceDirection: 'vertical' }},
          width: 1.5
        }},
        physics: false,
        interaction: {{ dragNodes: true, zoomView: true, dragView: true }}
      }};
      new vis.Network(container, data, options);
    </script>
    """


# ---------------------------------------------------------------------------
# Sidebar — uploads, document list, AI Tools nav, settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Research Assistant")
    st.caption(f"Session: `{st.session_state.session_id}`")

    st.markdown("### 📂 Upload Knowledge Sources")
    with st.container(border=True):
        st.markdown("**Supported files**")
        st.markdown(
            "📄 PDF · 📃 DOCX · 📽️ PPTX · 📊 XLSX · 📈 CSV  \n"
            "📝 TXT / Markdown · 💻 Code files  \n"
            "🖼️ Images (JPG, PNG, WEBP, BMP, TIFF)"
        )
        uploader_types = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
        uploaded_files = st.file_uploader(
            "Drag & drop files here, or click to browse",
            type=uploader_types, accept_multiple_files=True,
        )
        st.caption("Max 200MB per file · images are described via a vision model (needs Groq)")

    if uploaded_files:
        for uf in uploaded_files:
            already_indexed = uf.name in vs.list_filenames()
            if already_indexed:
                continue
            with st.spinner(f"Indexing {uf.name}..."):
                persistent_path = _uploaded_file_path(st.session_state.session_id, uf.name)
                persistent_path.parent.mkdir(parents=True, exist_ok=True)
                persistent_path.write_bytes(uf.getvalue())
                try:
                    num_chunks, num_pages = vs.add_document(str(persistent_path), uf.name)
                    if num_chunks == 0:
                        st.warning(f"{uf.name}: no extractable text found — skipped.")
                        persistent_path.unlink(missing_ok=True)
                    else:
                        db.add_document(uf.name, num_pages, num_chunks, st.session_state.session_id)
                        st.success(f"{uf.name}: {num_pages} page(s), {num_chunks} chunks indexed")
                except (DocumentLoadError, LLMError) as e:
                    st.error(f"{uf.name}: {e}")
                    persistent_path.unlink(missing_ok=True)
                except Exception:
                    persistent_path.unlink(missing_ok=True)
                    raise

    docs = db.get_documents(st.session_state.session_id)
    st.markdown(f"### 📁 Indexed Documents ({len(docs)})")
    if docs:
        with st.container(border=True):
            for d in docs:
                st.markdown(f"📄 **{d['filename']}**")
                st.caption(f"{d['num_pages']} pages · {d['num_chunks']} chunks")
    else:
        st.caption("No documents yet. Upload a PDF above to get started.")

    st.divider()
    st.subheader("🧰 AI Tools")
    tool_labels = [t["label"] for t in TOOLS]
    selected_label = st.radio(
        "Choose a tool", tool_labels, key="active_tool_label", label_visibility="collapsed"
    )
    active_tool = next(t["key"] for t in TOOLS if t["label"] == selected_label)

    st.divider()

    groq_ready = bool(os.environ.get("GROQ_API_KEY"))
    hf_ready = bool(os.environ.get("HF_API_TOKEN"))

    st.subheader("LLM provider")
    status_lines = [
        f"{'✅' if groq_ready else '⬜'} Groq (`GROQ_API_KEY`)",
        f"{'✅' if hf_ready else '⬜'} Hugging Face (`HF_API_TOKEN`)",
    ]
    st.caption("  \n".join(status_lines))

    if not (groq_ready or hf_ready):
        st.warning(
            "⚠️ No LLM provider available. Set `GROQ_API_KEY` (free, fastest — "
            "console.groq.com) or `HF_API_TOKEN` below to get started.",
            icon="⚠️",
        )

    with st.expander("🔑 Save API key" if not (groq_ready or hf_ready) else "🔑 Update API key"):
        st.caption(
            "Saved to a local `.env` file in the project folder — loads automatically "
            "every time you run the app, no need to set it in the terminal again."
        )
        groq_input = st.text_input(
            "Groq API key", type="password", placeholder="gsk_...",
            help="Get a free key at console.groq.com/keys",
        )
        hf_input = st.text_input(
            "Hugging Face token (optional)", type="password", placeholder="hf_...",
            help="Get a free token at huggingface.co/settings/tokens",
        )
        if st.button("💾 Save", use_container_width=True):
            saved_any = False
            if groq_input.strip():
                set_key(str(ENV_PATH), "GROQ_API_KEY", groq_input.strip())
                os.environ["GROQ_API_KEY"] = groq_input.strip()
                saved_any = True
            if hf_input.strip():
                set_key(str(ENV_PATH), "HF_API_TOKEN", hf_input.strip())
                os.environ["HF_API_TOKEN"] = hf_input.strip()
                saved_any = True
            if saved_any:
                st.success("Saved! This key will now load automatically every time.")
                st.rerun()
            else:
                st.warning("Enter at least one key before saving.")

    top_k = st.slider("Chunks to retrieve per question", 3, 10, 5)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear session", use_container_width=True):
            vs.clear()
            db.clear_session(st.session_state.session_id)
            st.session_state.chat_history = []
            session_upload_dir = UPLOADS_DIR / st.session_state.session_id
            if session_upload_dir.exists():
                import shutil
                shutil.rmtree(session_upload_dir, ignore_errors=True)
            st.rerun()
    with col_b:
        if st.button("📄 Export PDF", use_container_width=True, disabled=not st.session_state.chat_history):
            path = export_chat_to_pdf(st.session_state.chat_history, st.session_state.session_id)
            with open(path, "rb") as f:
                st.download_button("⬇️ Download", f, file_name=Path(path).name, mime="application/pdf")

# ---------------------------------------------------------------------------
# Main area — renders exactly one tool, based on the sidebar selection
# ---------------------------------------------------------------------------
active_meta = next(t for t in TOOLS if t["key"] == active_tool)
st.title(active_meta["label"])
st.caption(active_meta["caption"])
st.divider()

# ============================== Dashboard ==============================
if active_tool == "dashboard":
    tool_counts = db.get_tool_usage_counts(st.session_state.session_id)

    total_pages = sum(d["num_pages"] or 0 for d in docs)
    elapsed = datetime.utcnow() - st.session_state.session_start
    total_seconds = int(elapsed.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m {seconds}s"

    with st.container(border=True):
        st.markdown("##### 📚 Content Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", len(docs))
        c2.metric("Pages Indexed", total_pages)
        c3.metric("Chunks Indexed", len(vs.chunks))
        c4.metric("Session Time", duration_str)

    with st.container(border=True):
        st.markdown("##### 💬 Engagement")
        c1, c2, c3 = st.columns(3)
        c1.metric("Questions Asked", len(st.session_state.chat_history))
        c2.metric("Voice Questions", tool_counts.get("voice", 0))
        c3.metric("Translations Run", tool_counts.get("translate", 0))

    with st.container(border=True):
        st.markdown("##### 🛠️ Tools Used")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🎯 Quizzes", tool_counts.get("quiz", 0))
        c2.metric("🧠 Mind Maps", tool_counts.get("mindmap", 0))
        c3.metric("⚖️ Comparisons", tool_counts.get("compare", 0))
        c4.metric("🔬 Deep Research", tool_counts.get("research", 0))
        c5.metric("🖼️ Images", tool_counts.get("image", 0))

    st.divider()
    if not docs:
        st.info("Upload a document from the sidebar to get started, then pick a tool on the left.")
    else:
        st.caption("Documents in this session:")
        for d in docs:
            st.markdown(f"- **{d['filename']}** — {d['num_pages']} page(s), {d['num_chunks']} chunks")

# ============================== Chat with PDF ==============================
elif active_tool == "chat":
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            db.clear_chat_history(st.session_state.session_id)
            st.rerun()

    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("sources"):
                with st.expander(f"📎 {len(entry['sources'])} source excerpt(s)"):
                    for s in entry["sources"]:
                        st.markdown(f"**{s['filename']}**, page {s['page']} — score {s['score']:.2f}")
                        st.markdown(f"> {s['text']}")

    question = st.chat_input("Ask a question about your uploaded papers...")

    if question:
        if not vs.has_documents():
            st.warning("Please upload at least one PDF first.")
        else:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    retrieved = vs.search(question, top_k=top_k)

                try:
                    answer = st.write_stream(generate_answer_stream(question, retrieved))
                except LLMError as e:
                    answer = f"⚠️ {e}"
                    st.markdown(answer)

                sources = [
                    {"filename": r.filename, "page": r.page, "score": r.score, "text": r.text}
                    for r in retrieved
                ]
                if sources:
                    with st.expander(f"📎 {len(sources)} source excerpt(s)"):
                        for s in sources:
                            st.markdown(f"**{s['filename']}**, page {s['page']} — score {s['score']:.2f}")
                            st.markdown(f"> {s['text']}")

            db.add_chat_entry(st.session_state.session_id, question, answer, sources)
            st.session_state.chat_history.append(
                {"question": question, "answer": answer, "sources": sources}
            )

# ============================== Quiz & Flashcards ==============================
elif active_tool == "quiz":
    if not docs:
        st.info("Upload a PDF first to use this tool.")
    else:
        quiz_doc = st.selectbox("Choose a document", [d["filename"] for d in docs], key="quiz_doc_select")
        col1, col2, col3 = st.columns(3)
        with col1:
            quiz_type = st.selectbox("Type", QUIZ_TYPES, key="quiz_type_select")
        with col2:
            quiz_difficulty = st.selectbox("Difficulty", DIFFICULTIES, index=1, key="quiz_difficulty_select")
        with col3:
            quiz_num = st.slider("Items per click", 3, 30, 10, key="quiz_num_select")

        st.caption(
            "Groq's free tier limits how much a single request can generate. Want 50+? "
            "Click Generate a few times with the same type/document — each click adds "
            "more instead of starting over."
        )

        col_gen, col_clear = st.columns([3, 1])
        with col_gen:
            generate_clicked = st.button("➕ Generate", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.pop("quiz_data", None)
                st.session_state.pop("quiz_checked", None)
                st.rerun()

        if generate_clicked:
            with st.spinner(f"Generating {quiz_num} {quiz_type.lower()}..."):
                matching_chunks = [c for c in vs.chunks if c.filename == quiz_doc]
                full_text = " ".join(c.text for c in matching_chunks)
                try:
                    new_items = generate_quiz(full_text, quiz_type, quiz_difficulty, quiz_num)
                    existing = st.session_state.get("quiz_data")
                    if existing and existing["type"] == quiz_type and existing["doc"] == quiz_doc:
                        existing["items"].extend(new_items)
                    else:
                        st.session_state.quiz_data = {"type": quiz_type, "items": new_items, "doc": quiz_doc}
                    db.log_tool_usage(st.session_state.session_id, "quiz")
                    st.session_state.pop("quiz_checked", None)
                except (LLMError, QuizGenerationError, ValueError) as e:
                    st.error(str(e))

        if st.session_state.get("quiz_data"):
            qdata = st.session_state.quiz_data
            qtype = qdata["type"]
            items = qdata["items"]
            st.markdown(f"**{len(items)} {qtype} — {qdata['doc']}**")

            if qtype == "MCQ":
                for i, item in enumerate(items):
                    st.markdown(f"**Q{i+1}. {item.get('question', '')}**")
                    st.radio(
                        f"mcq_{i}", item.get("options", []), key=f"quiz_mcq_choice_{i}",
                        label_visibility="collapsed",
                    )

                if st.button("✅ Check answers"):
                    st.session_state.quiz_checked = True

                if st.session_state.get("quiz_checked"):
                    score = 0
                    for i, item in enumerate(items):
                        options = item.get("options", [])
                        correct_idx = item.get("correct_index", -1)
                        correct_answer = options[correct_idx] if 0 <= correct_idx < len(options) else None
                        picked = st.session_state.get(f"quiz_mcq_choice_{i}")
                        is_correct = picked == correct_answer
                        score += int(is_correct)
                        icon = "✅" if is_correct else "❌"
                        st.markdown(f"{icon} **Q{i+1}:** correct answer — {correct_answer}")
                        if item.get("explanation"):
                            st.caption(item["explanation"])
                    st.info(f"Score: {score} / {len(items)}")

            elif qtype == "Flashcards":
                for item in items:
                    with st.expander(f"🃏 {item.get('front', '')}"):
                        st.markdown(item.get("back", ""))

            elif qtype == "Coding Questions":
                for i, item in enumerate(items):
                    with st.expander(f"💻 Q{i+1}. {item.get('question', '')}"):
                        if item.get("hint"):
                            st.markdown(f"**Hint:** {item['hint']}")
                        if item.get("sample_approach"):
                            st.markdown(f"**Sample approach:** {item['sample_approach']}")

            elif qtype == "Interview Questions":
                for i, item in enumerate(items):
                    with st.expander(f"🎤 Q{i+1}. {item.get('question', '')}"):
                        if item.get("what_to_cover"):
                            st.markdown(f"**What a strong answer covers:** {item['what_to_cover']}")

# ============================== Mind Map ==============================
elif active_tool == "mindmap":
    mm_topic = st.text_input(
        "Topic or question", placeholder="e.g. Explain Machine Learning",
        key="mindmap_topic_input",
    )
    doc_options = ["(none — use general knowledge)"] + [d["filename"] for d in docs]
    mm_doc_choice = st.selectbox(
        "Base it on a document (optional)", doc_options, key="mindmap_doc_select",
        help="Leave as 'none' to map out any topic from general knowledge, or pick a "
             "document to map its actual content.",
    )

    if st.button("Generate Mind Map"):
        with st.spinner("Building the mind map..."):
            source_text = None
            if mm_doc_choice != doc_options[0]:
                matching_chunks = [c for c in vs.chunks if c.filename == mm_doc_choice]
                source_text = " ".join(c.text for c in matching_chunks)
            try:
                tree = generate_mindmap(mm_topic, source_text=source_text)
                st.session_state.mindmap_data = tree
                db.log_tool_usage(st.session_state.session_id, "mindmap")
            except (ValueError, LLMError, MindMapError) as e:
                st.error(str(e))
                st.session_state.pop("mindmap_data", None)

    if st.session_state.get("mindmap_data"):
        if st.button("🗑️ Clear mind map"):
            st.session_state.pop("mindmap_data", None)
            st.rerun()
        tree = st.session_state.mindmap_data
        st.iframe(src=_render_mindmap_html(tree), height=540)
        with st.expander("📋 View as outline"):
            st.markdown(render_outline(tree))

# ============================== Compare Documents ==============================
elif active_tool == "compare":
    if len(docs) < 2:
        st.info(f"Upload at least 2 PDFs to compare documents. Currently: {len(docs)}.")
    else:
        doc_names = [d["filename"] for d in docs]
        col1, col2 = st.columns(2)
        with col1:
            cmp_doc_a = st.selectbox("Document A", doc_names, key="cmp_doc_a_select")
        with col2:
            non_a_indices = [i for i, name in enumerate(doc_names) if name != cmp_doc_a]
            default_b_index = non_a_indices[0] if non_a_indices else 0
            cmp_doc_b = st.selectbox("Document B", doc_names, key="cmp_doc_b_select", index=default_b_index)

        if cmp_doc_a == cmp_doc_b:
            st.caption("⚠️ Pick two different documents to compare.")

        if st.button("Compare Documents", disabled=(cmp_doc_a == cmp_doc_b)):
            with st.spinner(f"Comparing {cmp_doc_a} vs {cmp_doc_b}..."):
                text_a = " ".join(c.text for c in vs.chunks if c.filename == cmp_doc_a)
                text_b = " ".join(c.text for c in vs.chunks if c.filename == cmp_doc_b)
                try:
                    result = compare_documents(text_a, cmp_doc_a, text_b, cmp_doc_b)
                    st.session_state.comparison_data = {
                        "result": result, "doc_a": cmp_doc_a, "doc_b": cmp_doc_b,
                    }
                    db.log_tool_usage(st.session_state.session_id, "compare")
                except (ValueError, LLMError, DocComparisonError) as e:
                    st.error(str(e))
                    st.session_state.pop("comparison_data", None)

        if st.session_state.get("comparison_data"):
            if st.button("🗑️ Clear comparison"):
                st.session_state.pop("comparison_data", None)
                st.rerun()
            cdata = st.session_state.comparison_data
            result = cdata["result"]
            doc_a, doc_b = cdata["doc_a"], cdata["doc_b"]
            st.markdown(f"**Comparing:** `{doc_a}` vs `{doc_b}`")

            if result["similarities"]:
                st.markdown("#### ✅ Similarities")
                for s in result["similarities"]:
                    st.markdown(f"- {s}")

            if result["differences"]:
                st.markdown("#### 🔀 Differences")
                def _esc(s):
                    return str(s).replace("|", "\\|").replace("\n", " ")
                table_lines = [f"| Aspect | {_esc(doc_a)} | {_esc(doc_b)} |", "|---|---|---|"]
                for d in result["differences"]:
                    table_lines.append(f"| {_esc(d['aspect'])} | {_esc(d['document_a'])} | {_esc(d['document_b'])} |")
                st.markdown("\n".join(table_lines))

            col_a2, col_b2 = st.columns(2)
            with col_a2:
                if result["missing_in_a"]:
                    st.markdown(f"#### 🚫 Missing from `{doc_a}`")
                    for m in result["missing_in_a"]:
                        st.markdown(f"- {m}")
            with col_b2:
                if result["missing_in_b"]:
                    st.markdown(f"#### 🚫 Missing from `{doc_b}`")
                    for m in result["missing_in_b"]:
                        st.markdown(f"- {m}")

            if result["risks"]:
                st.markdown("#### ⚠️ Flagged Risks")
                severity_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                for r in result["risks"]:
                    icon = severity_icon.get(r["severity"], "🟡")
                    st.markdown(f"{icon} **{r['severity']}** (applies to: {r['document']}) — {r['description']}")

# ============================== Deep Research ==============================
elif active_tool == "research":
    if not docs:
        st.info("Upload a PDF first to use this tool.")
    else:
        st.caption(
            "Breaks your question into a search plan, retrieves evidence for each "
            "sub-question separately, reasons step by step, then answers with an "
            "honest confidence score — slower than regular chat, but more thorough."
        )
        dr_question = st.text_area(
            "Research question", placeholder="e.g. What are the strengths and weaknesses of this approach?",
            key="deep_research_question", height=80,
        )
        dr_num_subqueries = st.slider("Number of sub-questions to investigate", 2, 6, 4, key="deep_research_num_subq")

        if st.button("🔬 Run Deep Research"):
            with st.spinner("Planning research approach..."):
                try:
                    dr_result = run_deep_research(dr_question, vs, num_subqueries=dr_num_subqueries)
                    st.session_state.deep_research_data = dr_result
                    db.log_tool_usage(st.session_state.session_id, "research")
                except (ValueError, LLMError, DeepResearchError) as e:
                    st.error(str(e))
                    st.session_state.pop("deep_research_data", None)

        if st.session_state.get("deep_research_data"):
            if st.button("🗑️ Clear research"):
                st.session_state.pop("deep_research_data", None)
                st.rerun()
            dr_data = st.session_state.deep_research_data

            st.markdown("#### 🗺️ Search Plan")
            for i, subq in enumerate(dr_data["search_plan"], start=1):
                st.markdown(f"{i}. {subq}")

            if dr_data["reasoning_steps"]:
                with st.expander("🧠 Step-by-step reasoning"):
                    for i, step in enumerate(dr_data["reasoning_steps"], start=1):
                        st.markdown(f"**Step {i}.** {step}")

            st.markdown("#### ✅ Answer")
            st.markdown(dr_data["answer"])

            conf = dr_data["confidence"]
            conf_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf["label"], "🟡")
            st.markdown(f"#### {conf_icon} Confidence: {conf['label']} ({conf['score']}/100)")
            if conf["justification"]:
                st.caption(conf["justification"])

            if dr_data["sources"]:
                with st.expander(f"📚 {len(dr_data['sources'])} source(s) consulted"):
                    for s in dr_data["sources"]:
                        subq_tags = ", ".join(s["subqueries"])
                        st.markdown(f"**[Source {s['index']}] {s['filename']}**, page {s['page']} — score {s['score']:.2f}")
                        st.caption(f"Found via: {subq_tags}")
                        st.markdown(f"> {s['text']}")

# ============================== Chat with Image ==============================
elif active_tool == "image":
    if not docs:
        st.info("Upload a PDF first to use this tool.")
    else:
        img_doc = st.selectbox("Choose a document", [d["filename"] for d in docs], key="img_doc_select")
        pdf_path = _uploaded_file_path(st.session_state.session_id, img_doc)

        if not pdf_path.exists():
            st.warning(
                "The original file for this document isn't available anymore (it was "
                "uploaded before this feature was added, or the session data was moved). "
                "Re-upload the PDF to use image analysis on it."
            )
        else:
            try:
                page_count = get_page_count(str(pdf_path))
            except PDFVisionError as e:
                page_count = None
                st.error(str(e))

            if page_count:
                img_page = st.number_input(
                    "Page number", min_value=1, max_value=page_count, value=1, key="img_page_select",
                )
                img_question = st.text_input(
                    "Ask something specific about this page (optional)",
                    placeholder="e.g. What was the peak value shown in the chart?",
                    key="img_question_input",
                )

                if st.button("🔍 Analyze"):
                    with st.spinner("Reading the page..."):
                        try:
                            result_text = describe_page(str(pdf_path), img_page, question=img_question)
                            st.session_state.image_analysis_data = result_text
                            db.log_tool_usage(st.session_state.session_id, "image")
                        except (PDFVisionError, LLMError) as e:
                            st.error(str(e))
                            st.session_state.pop("image_analysis_data", None)

                if st.session_state.get("image_analysis_data"):
                    if st.button("🗑️ Clear analysis"):
                        st.session_state.pop("image_analysis_data", None)
                        st.rerun()
                    st.markdown("#### 🔎 Analysis")
                    st.markdown(st.session_state.image_analysis_data)

# ============================== Translate ==============================
elif active_tool == "translate":
    COMMON_LANGUAGES = [
        "Spanish", "French", "German", "Hindi", "Chinese (Simplified)", "Japanese",
        "Arabic", "Portuguese", "Russian", "Korean", "Italian", "Other (type below)",
    ]
    source_mode = st.radio(
        "What do you want to translate?", ["Paste text", "An uploaded document"],
        key="translate_source_mode",
    )

    text_to_translate = ""
    if source_mode == "Paste text":
        text_to_translate = st.text_area("Text to translate", height=150, key="translate_input_text")
    elif not docs:
        st.info("Upload a document first, or switch to 'Paste text' above.")
    else:
        chosen_doc = st.selectbox("Choose a document", [d["filename"] for d in docs], key="translate_doc_select")
        matching_chunks = [c for c in vs.chunks if c.filename == chosen_doc]
        text_to_translate = " ".join(c.text for c in matching_chunks)

    lang_choice = st.selectbox("Target language", COMMON_LANGUAGES, key="translate_lang_select")
    if lang_choice == "Other (type below)":
        lang_choice = st.text_input("Type the target language", key="translate_lang_custom")

    if st.button("Translate"):
        with st.spinner("Translating..."):
            try:
                result = translate_text(text_to_translate, lang_choice)
                st.session_state.translate_data = result
                db.log_tool_usage(st.session_state.session_id, "translate")
            except (ValueError, LLMError) as e:
                st.error(str(e))
                st.session_state.pop("translate_data", None)

    if st.session_state.get("translate_data"):
        if st.button("🗑️ Clear translation"):
            st.session_state.pop("translate_data", None)
            st.rerun()
        st.markdown("#### 🌍 Translation")
        st.markdown(st.session_state.translate_data)

# ============================== Voice Chat ==============================
elif active_tool == "voice":
    st.caption(
        "Speech-to-text uses Groq's free Whisper API (no extra cost). The optional "
        "'read answer aloud' below uses Groq's Orpheus text-to-speech, which — unlike "
        "the other features in this app — is a paid-per-character feature with no "
        "confirmed free tier. It's off by default."
    )

    audio_value = st.audio_input("🎤 Record your question")

    if audio_value is not None:
        import hashlib
        audio_bytes = audio_value.getvalue()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if st.session_state.get("voice_last_audio_hash") != audio_hash:
            with st.spinner("Transcribing..."):
                try:
                    transcript = transcribe_audio(audio_bytes, "recording.wav")
                    st.session_state.voice_transcript = transcript
                    st.session_state.voice_last_audio_hash = audio_hash
                except LLMError as e:
                    st.error(str(e))
                    st.session_state.pop("voice_transcript", None)

    if st.session_state.get("voice_transcript"):
        if st.button("🗑️ Clear recording"):
            st.session_state.pop("voice_transcript", None)
            st.session_state.pop("voice_last_audio_hash", None)
            st.rerun()
        edited_question = st.text_area(
            "Transcribed question (edit if needed before sending)",
            value=st.session_state.voice_transcript, key="voice_edited_question",
        )

        read_aloud = st.checkbox(
            "🔊 Read the answer aloud (Groq Orpheus TTS — small per-character cost)",
            value=False, key="voice_read_aloud",
        )
        voice_choice = None
        if read_aloud:
            voice_choice = st.selectbox("Voice", sorted(TTS_VALID_VOICES), key="voice_choice_select")

        if st.button("📤 Send"):
            if not vs.has_documents():
                st.warning("Please upload at least one PDF first.")
            elif not edited_question.strip():
                st.warning("The transcribed question is empty — try recording again.")
            else:
                with st.spinner("Searching documents..."):
                    retrieved = vs.search(edited_question, top_k=top_k)
                try:
                    answer = st.write_stream(generate_answer_stream(edited_question, retrieved))
                except LLMError as e:
                    answer = f"⚠️ {e}"
                    st.markdown(answer)

                sources = [
                    {"filename": r.filename, "page": r.page, "score": r.score, "text": r.text}
                    for r in retrieved
                ]
                if sources:
                    with st.expander(f"📎 {len(sources)} source excerpt(s)"):
                        for s in sources:
                            st.markdown(f"**{s['filename']}**, page {s['page']} — score {s['score']:.2f}")
                            st.markdown(f"> {s['text']}")

                db.add_chat_entry(st.session_state.session_id, edited_question, answer, sources)
                st.session_state.chat_history.append(
                    {"question": edited_question, "answer": answer, "sources": sources}
                )
                db.log_tool_usage(st.session_state.session_id, "voice")

                if read_aloud and not answer.startswith("⚠️"):
                    with st.spinner("Generating speech..."):
                        try:
                            audio_out = synthesize_speech(
                                answer, voice=voice_choice, max_total_chars=TTS_DEFAULT_MAX_TOTAL_CHARS,
                            )
                            st.audio(audio_out, format="audio/wav")
                            if len(answer) > TTS_DEFAULT_MAX_TOTAL_CHARS:
                                st.caption(
                                    f"Only the first {TTS_DEFAULT_MAX_TOTAL_CHARS} characters of the "
                                    f"answer were read aloud, to control cost and API usage."
                                )
                        except (ValueError, LLMError) as e:
                            st.error(str(e))