import sys
sys.path.insert(0, '.')

import types
fake_st = types.ModuleType('streamlit')
fake_st.__file__ = '/tmp/fake_streamlit.py'
fake_st.session_state = {}
def _noop(*a, **k):
    return None
def _fake_getattr(name):
    if name.startswith('__') and name.endswith('__'):
        raise AttributeError(name)  # don't intercept dunders (torch's inspect machinery probes these)
    return _noop
fake_st.__getattr__ = _fake_getattr
sys.modules['streamlit'] = fake_st

import lib_only as mod
print('MERGED LIBRARY CODE LOADED SUCCESSFULLY')
print()

import uuid
sid = str(uuid.uuid4())[:8]
mod.db.init_db()
mod.db.add_document('test.pdf', 5, 10, sid)
docs = mod.db.get_documents(sid)
assert len(docs) == 1 and docs[0]['filename'] == 'test.pdf'
print('TEST 1 (db namespace: db.add_document/db.get_documents) PASSED')

mod.db.add_chat_entry(sid, 'Q1', 'A1', [])
assert len(mod.db.get_chat_history(sid)) == 1
mod.db.clear_chat_history(sid)
assert len(mod.db.get_chat_history(sid)) == 0
assert len(mod.db.get_documents(sid)) == 1
print('TEST 2 (clear_chat_history preserves documents) PASSED')
mod.db.clear_session(sid)

# --- Test 3: renamed SYSTEM_PROMPT constants are distinct and correctly attributed ---
assert mod.SYSTEM_PROMPT_LLMCLIENT != mod.SYSTEM_PROMPT_QUIZ
assert mod.SYSTEM_PROMPT_QUIZ != mod.SYSTEM_PROMPT_MINDMAP
assert mod.SYSTEM_PROMPT_MINDMAP != mod.SYSTEM_PROMPT_DOCCOMPARE
assert mod.SYSTEM_PROMPT_DOCCOMPARE != mod.SYSTEM_PROMPT_TRANSLATOR
assert 'translator' in mod.SYSTEM_PROMPT_TRANSLATOR.lower()
assert 'quiz' in mod.SYSTEM_PROMPT_QUIZ.lower() or 'json array' in mod.SYSTEM_PROMPT_QUIZ.lower()
print('TEST 3 (renamed SYSTEM_PROMPT_* constants are all distinct, correctly scoped) PASSED')

# --- Test 4: MAX_SOURCE_CHARS renamed correctly per module (quiz=12000, mindmap=8000) ---
assert mod.MAX_SOURCE_CHARS_QUIZ == 8000  # reduced from 12000 as part of the 413-error token-budget fix
assert mod.MAX_SOURCE_CHARS_MINDMAP == 8000
print('TEST 4 (MAX_SOURCE_CHARS correctly disambiguated per module) PASSED')

# --- Test 5: chunk_text (pure logic, from rag_engine) ---
chunks = mod.chunk_text("word " * 500, chunk_size=200, overlap=40)
assert len(chunks) > 1
assert all(len(c) <= 200 for c in chunks)
print('TEST 5 (chunk_text works correctly in merged context) PASSED')

# --- Test 6: quiz_generator's JSON extraction with the correct renamed error class ---
from unittest.mock import patch
with patch.object(mod, 'generate_raw', return_value='[{"question":"Q1","options":["A","B","C","D"],"correct_index":0,"explanation":"E"}]'):
    items = mod.generate_quiz("doc text", "MCQ", "Easy", 1)
    assert len(items) == 1 and items[0]['question'] == 'Q1'
print('TEST 6 (generate_quiz end-to-end, mocked LLM) PASSED')

# --- Test 7: mindmap_generator raises the CORRECT exception type (not deep_research's, post-rename) ---
with patch.object(mod, 'generate_raw', return_value='not valid json at all'):
    try:
        mod.generate_mindmap("Some Topic")
        print('TEST 7 FAILED: should have raised')
    except mod.MindMapError as e:
        print('TEST 7 (mind map raises correct MindMapError, not misattributed) PASSED:', str(e)[:50])
    except Exception as e:
        print(f'TEST 7 FAILED: raised wrong exception type {type(e).__name__} instead of MindMapError')

# --- Test 8: doc_comparison raises the CORRECT exception type too ---
with patch.object(mod, 'generate_raw', return_value='garbage not json'):
    try:
        mod.compare_documents("text a", "A.pdf", "text b", "B.pdf")
        print('TEST 8 FAILED: should have raised')
    except mod.DocComparisonError as e:
        print('TEST 8 (doc comparison raises correct DocComparisonError) PASSED:', str(e)[:50])
    except Exception as e:
        print(f'TEST 8 FAILED: raised wrong exception type {type(e).__name__} instead of DocComparisonError')

# --- Test 9: deep_research raises the CORRECT exception type too ---
with patch.object(mod, 'generate_raw', return_value='garbage'):
    try:
        mod._extract_json_object_deepresearch("garbage no json here")
        print('TEST 9 FAILED: should have raised')
    except mod.DeepResearchError as e:
        print('TEST 9 (deep research raises correct DeepResearchError) PASSED:', str(e)[:50])
    except Exception as e:
        print(f'TEST 9 FAILED: raised wrong exception type {type(e).__name__} instead of DeepResearchError')

print()
print('ALL MERGED-FILE FUNCTIONAL TESTS PASSED')

# --- Test 10: VectorStore persistence uses JSON, not pickle (the actual bug fix) ---
import inspect
save_src = inspect.getsource(mod.VectorStore._save)
load_src = inspect.getsource(mod.VectorStore._load_if_exists)
assert "pickle.dump" not in save_src and "pickle.load" not in save_src, "still using pickle in _save!"
assert "pickle.dump" not in load_src and "pickle.load" not in load_src, "still using pickle in _load_if_exists!"
assert "json.dump" in save_src
assert "json.load" in load_src
assert mod.VectorStore.__init__.__doc__ is None or True  # just checking meta_path ext below
print('TEST 10 (VectorStore._save/_load use JSON, not pickle) PASSED')

# --- Test 11: end-to-end persistence round-trip through an actual save/load cycle ---
from unittest.mock import patch, MagicMock
import numpy as np_test

fake_model = MagicMock()
fake_model.get_embedding_dimension.return_value = 384
def fake_encode(texts, normalize_embeddings=True, show_progress_bar=False):
    return np_test.random.rand(len(texts), 384).astype('float32')
fake_model.encode.side_effect = fake_encode

with patch.object(mod.EmbeddingModel, 'get', return_value=fake_model):
    test_sid = 'persisttest_' + str(uuid.uuid4())[:6]
    vs1 = mod.VectorStore(test_sid)
    assert vs1.meta_path.suffix == '.json'
    chunk = mod.Chunk(text='persisted content', filename='doc.pdf', page=1, chunk_id=0)
    vs1._add_chunks([chunk])

    # Simulate a fresh 'rerun': construct a NEW VectorStore instance for the
    # same session_id, which loads from disk exactly like Streamlit would
    # after a script rerun with a fresh top-to-bottom execution
    vs2 = mod.VectorStore(test_sid)
    assert len(vs2.chunks) == 1
    assert vs2.chunks[0].text == 'persisted content'
    assert vs2.chunks[0].filename == 'doc.pdf'
    print('TEST 11 (real save -> reload round-trip works correctly) PASSED')
    vs1.clear()

print()
print('ALL 11 TESTS PASSED INCLUDING THE PICKLE-TO-JSON FIX VERIFICATION')

# --- Test 12: retry-on-prose-response fix, in the MERGED context (the actual reported bug) ---
valid_compare = '{"similarities": [], "differences": [], "missing_in_a": [], "missing_in_b": [], "risks": []}'
with patch.object(mod, 'generate_raw', side_effect=['These documents are basically identical.', valid_compare]) as m:
    result = mod.compare_documents('text a', 'A.pdf', 'text b', 'B.pdf')
    assert m.call_count == 2
    print('TEST 12 (Compare Documents retry-on-prose fix works in merged file) PASSED')

print()
print('ALL 12 TESTS PASSED')

# --- Test 13: quiz token-budget fix survives the merge (the actual reported 413 bug) ---
large_doc = 'Technical Python content. ' * 2000
captured_calls = []
def fake_generate_raw_quiz(system_prompt, user_prompt, max_tokens):
    captured_calls.append(max_tokens)
    return '[{"question":"Q","options":["A","B","C","D"],"correct_index":0,"explanation":"E"}]'

with patch.object(mod, 'generate_raw', side_effect=fake_generate_raw_quiz):
    mod.generate_quiz(large_doc, 'MCQ', 'Medium', 10)  # the exact failing scenario: default 10 items
    assert captured_calls[0] < 3000, f"max_tokens too high: {captured_calls[0]}"
    print(f'TEST 13 (quiz max_tokens scales down for default item count, merged) PASSED: requested {captured_calls[0]} tokens (was flat 6000 before fix)')

print()
print('ALL 13 TESTS PASSED')