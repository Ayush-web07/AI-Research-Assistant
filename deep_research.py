"""
deep_research.py
"Deep Research" mode: instead of answering a question directly from one retrieval
pass, this module breaks it into a search plan, retrieves for each sub-question
separately, then reasons step-by-step over the combined evidence before producing
a final answer with an honest confidence score.

Pipeline:
    Question -> generate_search_plan() -> [sub-questions]
             -> search vector store per sub-question -> deduped source list
             -> _synthesize() -> reasoning steps + answer + confidence
"""

import json
import re

from json_repair import repair_json

from llm_client import generate_raw

MAX_SUBQUERIES = 6
DEFAULT_SUBQUERIES = 4
DEFAULT_TOP_K_PER_SUBQUERY = 3

VALID_CONFIDENCE_LABELS = {"Low", "Medium", "High"}

PLAN_SYSTEM_PROMPT = (
    "You are a research planning assistant. You output ONLY a valid JSON array of "
    "strings, nothing else — no explanations, no markdown, no code fences."
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a rigorous deep-research assistant. Reason step by step using ONLY the "
    "provided source excerpts before answering. You output ONLY a valid JSON object, "
    "nothing else — no explanations, no markdown, no code fences, no text before or "
    "after the JSON."
)


class DeepResearchError(Exception):
    """Raised when a stage of the pipeline produces output that can't be parsed/validated."""
    pass


# ---------------------------------------------------------------------------
# JSON parsing helpers (duplicated intentionally, mirrors the pattern used in
# quiz_generator.py / mindmap_generator.py / doc_comparison.py — keeps each
# feature module independent so a bug in one can't ripple into the others)
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> list:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise DeepResearchError(
            "The model's response didn't contain a recognizable JSON array."
        )
    json_str = text[start:end + 1]
    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(repair_json(json_str), strict=False)
    except (json.JSONDecodeError, Exception) as e:
        raise DeepResearchError(f"Couldn't parse the search plan as JSON, even after attempting repair: {e}")


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise DeepResearchError(
            "The model's response didn't contain a recognizable JSON object."
        )
    json_str = text[start:end + 1]
    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(repair_json(json_str), strict=False)
    except (json.JSONDecodeError, Exception) as e:
        raise DeepResearchError(f"Couldn't parse the synthesis result as JSON, even after attempting repair: {e}")


# ---------------------------------------------------------------------------
# Stage 1: search plan
# ---------------------------------------------------------------------------

def generate_search_plan(question: str, num_subqueries: int = DEFAULT_SUBQUERIES) -> list:
    """Break the question into a handful of focused sub-questions/search queries
    that together would help answer it more thoroughly than a single retrieval pass."""
    if not question or not question.strip():
        raise ValueError("A research question is required.")
    num_subqueries = max(1, min(num_subqueries, MAX_SUBQUERIES))

    user_prompt = (
        f"Break this research question down into exactly {num_subqueries} focused "
        f'sub-questions or search queries that together would help answer it '
        f'thoroughly:\n\n"{question}"\n\n'
        "Respond with ONLY a JSON array of strings, no prose, no code fences."
    )
    raw = generate_raw(PLAN_SYSTEM_PROMPT, user_prompt, max_tokens=600)
    items = _extract_json_array(raw)
    plan = [str(x).strip() for x in items if isinstance(x, (str, int, float)) and str(x).strip()]
    if not plan:
        raise DeepResearchError("The model returned an empty search plan.")
    return plan


# ---------------------------------------------------------------------------
# Stage 2: multi-query retrieval + dedup (pure logic, no LLM call — testable
# without mocking anything beyond a fake vector store)
# ---------------------------------------------------------------------------

def retrieve_for_plan(vector_store, plan: list, top_k_per_subquery: int = DEFAULT_TOP_K_PER_SUBQUERY) -> list:
    """Search the vector store once per sub-question in the plan, and merge results
    into a deduplicated, numbered source list. The same chunk surfaced by multiple
    sub-questions is recorded once, with all the sub-questions that found it."""
    seen = {}
    order = []
    for subq in plan:
        results = vector_store.search(subq, top_k=top_k_per_subquery)
        for r in results:
            dedup_key = (r.filename, r.page, r.text)
            if dedup_key not in seen:
                seen[dedup_key] = {
                    "filename": r.filename, "page": r.page, "text": r.text,
                    "score": r.score, "subqueries": [subq],
                }
                order.append(dedup_key)
            elif subq not in seen[dedup_key]["subqueries"]:
                seen[dedup_key]["subqueries"].append(subq)

    sources = [seen[k] for k in order]
    for i, s in enumerate(sources, start=1):
        s["index"] = i
    return sources


def _build_numbered_context(sources: list) -> str:
    parts = []
    for s in sources:
        parts.append(f"[Source {s['index']}] ({s['filename']}, page {s['page']}):\n{s['text']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 3: step-by-step synthesis with confidence scoring
# ---------------------------------------------------------------------------

def _validate_confidence(raw_confidence) -> dict:
    if not isinstance(raw_confidence, dict):
        raw_confidence = {}
    try:
        score = int(round(float(raw_confidence.get("score", 50))))
    except (TypeError, ValueError):
        score = 50
    score = max(0, min(100, score))

    label = str(raw_confidence.get("label", "")).strip().capitalize()
    if label not in VALID_CONFIDENCE_LABELS:
        # derive a sensible label from the score if the model didn't give a valid one
        label = "High" if score >= 75 else "Medium" if score >= 40 else "Low"

    justification = str(raw_confidence.get("justification", "")).strip()
    return {"score": score, "label": label, "justification": justification}


def _synthesize(question: str, plan: list, sources: list) -> dict:
    context = _build_numbered_context(sources)
    plan_text = "\n".join(f"- {p}" for p in plan)
    user_prompt = (
        f"Original question: {question}\n\n"
        f"Research plan (sub-questions investigated):\n{plan_text}\n\n"
        f"Source excerpts:\n\n{context}\n\n"
        "Reason through this step by step across the sub-questions, then provide a "
        "final answer. Cite sources inline as [Source N]. Assess your confidence "
        "honestly — lower it if sources conflict, are incomplete, or are only "
        "tangentially relevant to the question.\n\n"
        "Respond with ONLY a valid JSON object matching this schema, no prose, no "
        "code fences:\n"
        '{"reasoning_steps": ["step 1 ...", "step 2 ..."], '
        '"answer": "... with [Source N] citations ...", '
        '"confidence": {"score": 0-100, "label": "Low|Medium|High", "justification": "..."}}'
    )
    raw = generate_raw(SYNTHESIS_SYSTEM_PROMPT, user_prompt, max_tokens=3200)
    data = _extract_json_object(raw)

    reasoning_steps = data.get("reasoning_steps", [])
    if not isinstance(reasoning_steps, list):
        reasoning_steps = []
    reasoning_steps = [str(s).strip() for s in reasoning_steps if str(s).strip()]

    answer = str(data.get("answer", "")).strip()
    if not answer:
        raise DeepResearchError("The model didn't return a final answer.")

    confidence = _validate_confidence(data.get("confidence"))

    return {"reasoning_steps": reasoning_steps, "answer": answer, "confidence": confidence}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_deep_research(
    question: str,
    vector_store,
    num_subqueries: int = DEFAULT_SUBQUERIES,
    top_k_per_subquery: int = DEFAULT_TOP_K_PER_SUBQUERY,
) -> dict:
    """Run the full deep research pipeline and return a structured result:

        {
            "search_plan": [str, ...],
            "reasoning_steps": [str, ...],
            "answer": str,
            "confidence": {"score": int, "label": str, "justification": str},
            "sources": [{"index": int, "filename": str, "page": int, "text": str,
                         "score": float, "subqueries": [str, ...]}, ...],
        }

    Raises:
        ValueError: empty question, or no documents indexed yet
        LLMError: no provider available or a provider call failed
        DeepResearchError: a pipeline stage's output couldn't be parsed/validated
    """
    if not question or not question.strip():
        raise ValueError("A research question is required.")
    if not vector_store.has_documents():
        raise ValueError("No documents indexed yet — upload at least one PDF first.")

    plan = generate_search_plan(question, num_subqueries)
    sources = retrieve_for_plan(vector_store, plan, top_k_per_subquery)

    if not sources:
        return {
            "search_plan": plan,
            "reasoning_steps": [],
            "answer": "No relevant content was found in the uploaded documents for this question.",
            "confidence": {
                "score": 0, "label": "Low",
                "justification": "No supporting evidence was retrieved from the uploaded documents.",
            },
            "sources": [],
        }

    synthesis = _synthesize(question, plan, sources)
    synthesis["search_plan"] = plan
    synthesis["sources"] = sources
    return synthesis