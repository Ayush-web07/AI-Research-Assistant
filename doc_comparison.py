"""
doc_comparison.py
Compares two documents (e.g. two contract versions, two research papers) and
returns a structured breakdown: similarities, differences, clauses/points
present in one but missing from the other, and flagged risks.
"""

import json
import re

from json_repair import repair_json

from llm_client import generate_raw

MAX_CHARS_PER_DOC = 6000  # keep combined prompt within free-tier context limits

SYSTEM_PROMPT = (
    "You are a meticulous document comparison assistant. You output ONLY a valid JSON "
    "object, nothing else — no explanations, no markdown formatting, no code fences, "
    "no text before or after the JSON."
)

SCHEMA_HINT = (
    '{"similarities": ["...", "..."], '
    '"differences": [{"aspect": "...", "document_a": "...", "document_b": "..."}], '
    '"missing_in_a": ["...", "..."], '
    '"missing_in_b": ["...", "..."], '
    '"risks": [{"description": "...", "severity": "Low|Medium|High", "document": "A|B|Both"}]}'
)

VALID_SEVERITIES = {"Low", "Medium", "High"}
VALID_DOC_TAGS = {"A", "B", "Both"}


class DocComparisonError(Exception):
    """Raised when the model's output can't be parsed/validated as a comparison result."""
    pass


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise DocComparisonError(
            "The model's response didn't contain a recognizable JSON object. Try again."
        )
    json_str = text[start:end + 1]
    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(repair_json(json_str), strict=False)
    except (json.JSONDecodeError, Exception) as e:
        raise DocComparisonError(
            f"Couldn't parse the model's output as JSON, even after attempting repair: {e}. Try again."
        )


def _as_str_list(value) -> list:
    if not isinstance(value, list):
        return []
    result = []
    for v in value:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            result.append(s)
    return result


def _validate_result(data: dict) -> dict:
    if not isinstance(data, dict):
        raise DocComparisonError("Malformed comparison result: expected a JSON object.")

    similarities = _as_str_list(data.get("similarities"))
    missing_in_a = _as_str_list(data.get("missing_in_a"))
    missing_in_b = _as_str_list(data.get("missing_in_b"))

    differences = []
    for d in data.get("differences", []) if isinstance(data.get("differences"), list) else []:
        if not isinstance(d, dict):
            continue
        aspect = str(d.get("aspect", "")).strip()
        if not aspect:
            continue
        differences.append({
            "aspect": aspect,
            "document_a": str(d.get("document_a", "")).strip(),
            "document_b": str(d.get("document_b", "")).strip(),
        })

    risks = []
    for r in data.get("risks", []) if isinstance(data.get("risks"), list) else []:
        if not isinstance(r, dict):
            continue
        description = str(r.get("description", "")).strip()
        if not description:
            continue
        severity = str(r.get("severity", "Medium")).strip().capitalize()
        if severity not in VALID_SEVERITIES:
            severity = "Medium"
        doc_tag = str(r.get("document", "Both")).strip().capitalize()
        if doc_tag not in VALID_DOC_TAGS:
            doc_tag = "Both"
        risks.append({"description": description, "severity": severity, "document": doc_tag})

    return {
        "similarities": similarities,
        "differences": differences,
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
        "risks": risks,
    }


def compare_documents(text_a: str, name_a: str, text_b: str, name_b: str) -> dict:
    """Compare two documents and return a structured breakdown.

    Returns a dict with keys: similarities (list[str]), differences
    (list[{aspect, document_a, document_b}]), missing_in_a (list[str]),
    missing_in_b (list[str]), risks (list[{description, severity, document}]).

    Raises:
        ValueError: either document text is empty
        LLMError: no provider available or the provider call failed
        DocComparisonError: the model's output couldn't be parsed/validated
    """
    if not text_a or not text_a.strip():
        raise ValueError(f"'{name_a}' has no extracted text to compare.")
    if not text_b or not text_b.strip():
        raise ValueError(f"'{name_b}' has no extracted text to compare.")

    truncated_a = text_a[:MAX_CHARS_PER_DOC]
    truncated_b = text_b[:MAX_CHARS_PER_DOC]

    user_prompt = (
        f"Document A ({name_a}):\n{truncated_a}\n\n"
        f"Document B ({name_b}):\n{truncated_b}\n\n"
        "Compare these two documents. Identify:\n"
        "1. Key similarities between them\n"
        "2. Key differences, aspect by aspect (e.g. terms, clauses, findings, values)\n"
        "3. Anything present in Document A but missing from Document B\n"
        "4. Anything present in Document B but missing from Document A\n"
        "5. Any risks worth flagging (e.g. unfavorable terms, contradictions, gaps, "
        "ambiguities), each with a severity (Low/Medium/High) and which document(s) "
        "it applies to (A, B, or Both)\n\n"
        f"Respond with ONLY a valid JSON object matching this schema, no prose, no "
        f"code fences:\n{SCHEMA_HINT}"
    )

    raw = generate_raw(SYSTEM_PROMPT, user_prompt, max_tokens=3800)
    data = _extract_json_object(raw)
    return _validate_result(data)