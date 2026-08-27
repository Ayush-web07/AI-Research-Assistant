"""
merge.py — combines the 14-file AI Research Assistant project into a single
app.py, stripping cross-module local imports (they're unnecessary once
everything lives in one namespace) while preserving all external library
imports and all functionality unchanged.
"""

import re
from pathlib import Path

SRC = Path("/home/claude/ai-research-assistant")
OUT = Path("/home/claude/single-file-build/app_single_file.py")

ORDER = [
    "database.py",
    "pdf_vision.py",
    "llm_client.py",
    "document_loaders.py",
    "rag_engine.py",
    "image_analysis.py",
    "quiz_generator.py",
    "mindmap_generator.py",
    "doc_comparison.py",
    "deep_research.py",
    "translator.py",
    "voice_client.py",
    "export_utils.py",
]

LOCAL_MODULES = {
    "database", "pdf_vision", "llm_client", "document_loaders", "rag_engine",
    "image_analysis", "quiz_generator", "mindmap_generator", "doc_comparison",
    "deep_research", "translator", "voice_client", "export_utils",
}

# Confirmed via AST-based collision detection (detect_collisions.py) — these
# top-level names are independently defined in multiple files with DIFFERENT
# content. Left unrenamed, a naive merge would silently make every module use
# whichever definition loads last (e.g. Chat's grounding SYSTEM_PROMPT could
# get overwritten by Translate's), with no error to signal it happened.
RENAMES = {
    "quiz_generator.py": {
        "MAX_SOURCE_CHARS": "MAX_SOURCE_CHARS_QUIZ",
        "SYSTEM_PROMPT": "SYSTEM_PROMPT_QUIZ",
    },
    "mindmap_generator.py": {
        "MAX_SOURCE_CHARS": "MAX_SOURCE_CHARS_MINDMAP",
        "SCHEMA_HINT": "SCHEMA_HINT_MINDMAP",
        "SYSTEM_PROMPT": "SYSTEM_PROMPT_MINDMAP",
        "_extract_json_object": "_extract_json_object_mindmap",
    },
    "doc_comparison.py": {
        "SCHEMA_HINT": "SCHEMA_HINT_DOCCOMPARE",
        "SYSTEM_PROMPT": "SYSTEM_PROMPT_DOCCOMPARE",
        "_extract_json_object": "_extract_json_object_doccompare",
    },
    "deep_research.py": {
        "_extract_json_object": "_extract_json_object_deepresearch",
    },
    "translator.py": {
        "SYSTEM_PROMPT": "SYSTEM_PROMPT_TRANSLATOR",
    },
    "llm_client.py": {
        "SYSTEM_PROMPT": "SYSTEM_PROMPT_LLMCLIENT",
    },
}


def apply_renames(fname: str, text: str) -> str:
    for old, new in RENAMES.get(fname, {}).items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def strip_local_imports(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"from (\w+) import ", stripped)
        if m and m.group(1) in LOCAL_MODULES:
            continue
        if stripped == "import database as db":
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def get_body(filename: str) -> str:
    text = (SRC / filename).read_text(encoding="utf-8")
    text = re.sub(r'^\s*"""[\s\S]*?"""\s*\n', "", text, count=1)
    text = apply_renames(filename, text)
    return strip_local_imports(text).strip("\n")


parts = []

parts.append('''"""
AI Research Assistant — single-file build.

This file combines the project's 14 modules (database, rag_engine, llm_client,
document_loaders, pdf_vision, image_analysis, quiz_generator, mindmap_generator,
doc_comparison, deep_research, translator, voice_client, export_utils, app) into
one file, so there is nothing to import between local modules and therefore no
way for cross-file copy/paste mistakes to cause circular-import or
missing-attribute errors.

Run with:
    streamlit run app_single_file.py
"""

import asyncio
import base64
import csv
import hashlib
import io
import json
import os
import pickle
import shutil
import sqlite3
import sys
import time
import types
import uuid
import warnings
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=FutureWarning)

import re

import requests
import streamlit as st
from dotenv import load_dotenv, set_key

import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

import fitz  # PyMuPDF
from PIL import Image

import docx  # python-docx
import openpyxl
from pptx import Presentation  # python-pptx

from json_repair import repair_json

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
''')

for fname in ORDER:
    parts.append(f"\n# {'=' * 76}\n# --- from {fname} " + "=" * max(0, 60 - len(fname)) + f"\n# {'=' * 76}\n")
    parts.append(get_body(fname))

parts.append('''

# ---------------------------------------------------------------------------
# Expose database.py's functions as db.<name>(...) so every db.xxx() call in
# the app section below (originally `import database as db`) keeps working
# unchanged, without needing a separate module.
# ---------------------------------------------------------------------------
db = types.SimpleNamespace(
    init_db=init_db,
    add_document=add_document,
    get_documents=get_documents,
    add_chat_entry=add_chat_entry,
    get_chat_history=get_chat_history,
    clear_chat_history=clear_chat_history,
    clear_session=clear_session,
    log_tool_usage=log_tool_usage,
    get_tool_usage_counts=get_tool_usage_counts,
)
''')

parts.append(f"\n# {'=' * 76}\n# --- from app.py (main Streamlit script) " + "=" * 30 + f"\n# {'=' * 76}\n")
app_text = (SRC / "app.py").read_text(encoding="utf-8")
app_text = re.sub(r'^\s*"""[\s\S]*?"""\s*\n', "", app_text, count=1)

marker = "ENV_PATH = Path(__file__).parent / \".env\""
idx = app_text.find(marker)
app_text = app_text[idx:]

parts.append(app_text)

OUT.write_text("\n".join(parts), encoding="utf-8")
print(f"Wrote {OUT}, {len(OUT.read_text().splitlines())} lines")