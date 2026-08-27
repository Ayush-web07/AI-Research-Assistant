"""
mindmap_generator.py
Generates a hierarchical mind map (topic -> subtopics -> sub-subtopics), either
from a free-typed topic (general knowledge, e.g. "Explain Machine Learning") or
grounded in an uploaded document's actual content.

Returns a nested dict tree; tree_to_graph() flattens it into nodes/edges for
rendering as an interactive graph, and render_outline() renders it as a plain
markdown fallback (useful if JS doesn't load, or for copy/paste).
"""

import json
import re

from json_repair import repair_json

from llm_client import generate_raw

MAX_SOURCE_CHARS = 8000

SYSTEM_PROMPT = (
    "You are a mind-map generation assistant. You output ONLY a valid JSON object "
    "representing a hierarchical tree, nothing else — no explanations, no markdown "
    "formatting, no code fences, no text before or after the JSON."
)

SCHEMA_HINT = (
    '{"topic": "Root Topic", "children": [{"topic": "Subtopic 1", "children": '
    '[{"topic": "Detail A", "children": []}, {"topic": "Detail B", "children": []}]}, '
    '{"topic": "Subtopic 2", "children": []}]}'
)


class MindMapError(Exception):
    """Raised when the model's output can't be parsed/validated as a mind map."""
    pass


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise MindMapError(
            "The model's response didn't contain a recognizable JSON object. "
            "Try again, possibly with a narrower topic."
        )
    json_str = text[start:end + 1]
    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(repair_json(json_str), strict=False)
    except (json.JSONDecodeError, Exception) as e:
        raise MindMapError(
            f"Couldn't parse the model's output as JSON, even after attempting repair: {e}. "
            f"Try again, possibly with a narrower topic."
        )


def _validate_tree(node, depth: int = 0, max_depth: int = 4) -> dict:
    """Structural validation + normalization: every node gets a string 'topic' and
    a list 'children', recursively. Caps runaway depth defensively."""
    if not isinstance(node, dict):
        raise MindMapError("Malformed mind map: expected an object at each node.")
    topic = str(node.get("topic", "")).strip()
    if not topic:
        raise MindMapError("Malformed mind map: a node is missing its 'topic'.")

    raw_children = node.get("children", [])
    if not isinstance(raw_children, list):
        raw_children = []

    children = []
    if depth < max_depth:
        for child in raw_children:
            children.append(_validate_tree(child, depth + 1, max_depth))

    return {"topic": topic, "children": children}


def generate_mindmap(topic: str, source_text: str = None, max_children: int = 5) -> dict:
    """Generate a hierarchical mind map.

    Args:
        topic: the subject to map out (e.g. "Machine Learning", or a question).
        source_text: optional grounding text (e.g. an uploaded document's content).
            If provided, the map is built from this content rather than general
            knowledge.
        max_children: soft cap on branches per node, passed to the model as guidance.

    Returns:
        A nested dict: {"topic": str, "children": [ ...same shape... ]}

    Raises:
        ValueError: empty/missing topic
        LLMError: no provider available or the provider call failed
        MindMapError: the model's output couldn't be parsed/validated
    """
    if not topic or not topic.strip():
        raise ValueError("A topic is required to generate a mind map.")

    if source_text:
        truncated = source_text[:MAX_SOURCE_CHARS]
        user_prompt = (
            f"Document content:\n\n{truncated}\n\n"
            f'Create a mind map of the key concepts in this document, organized around '
            f'the theme: "{topic}". Use a root topic, up to {max_children} main branches, '
            f"and up to {max_children} sub-branches under each, going up to 3 levels deep. "
            f"Base it strictly on the document content.\n\n"
            f"Respond with ONLY a valid JSON object matching this schema, no prose, no "
            f"code fences:\n{SCHEMA_HINT}"
        )
    else:
        user_prompt = (
            f'Create a mind map explaining: "{topic}". Use a root topic, up to '
            f"{max_children} main branches, and up to {max_children} sub-branches under "
            f"each, going up to 3 levels deep.\n\n"
            f"Respond with ONLY a valid JSON object matching this schema, no prose, no "
            f"code fences:\n{SCHEMA_HINT}"
        )

    raw = generate_raw(SYSTEM_PROMPT, user_prompt, max_tokens=2500)
    tree = _extract_json_object(raw)
    return _validate_tree(tree)


def tree_to_graph(tree: dict):
    """Flatten a mind-map tree into (nodes, edges) lists suitable for a graph
    visualization library.

    nodes: [{"id": int, "label": str, "level": int}]
    edges: [{"from": int, "to": int}]
    """
    nodes = []
    edges = []
    counter = {"next_id": 0}

    def walk(node, parent_id, level):
        my_id = counter["next_id"]
        counter["next_id"] += 1
        nodes.append({"id": my_id, "label": node["topic"], "level": level})
        if parent_id is not None:
            edges.append({"from": parent_id, "to": my_id})
        for child in node.get("children", []):
            walk(child, my_id, level + 1)

    walk(tree, None, 0)
    return nodes, edges


def render_outline(tree: dict, indent: int = 0) -> str:
    """Render the tree as a markdown bullet-list outline — a plain-text fallback
    view alongside the interactive graph (also handy for copy/paste)."""
    bullet = "  " * indent + f"- {tree['topic']}"
    lines = [bullet]
    for child in tree.get("children", []):
        lines.append(render_outline(child, indent + 1))
    return "\n".join(lines)