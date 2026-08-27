"""
quiz_generator.py
Generates practice quizzes from an indexed document's text: MCQs, coding
questions, interview questions, or flashcards, at a chosen difficulty level.

Asks the LLM for strict JSON output and parses it defensively, since free-tier
models occasionally wrap JSON in markdown code fences or add stray prose.
"""

import json
import re

from json_repair import repair_json

from llm_client import generate_raw, LLMError

QUIZ_TYPES = ["MCQ", "Coding Questions", "Interview Questions", "Flashcards"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]

MAX_SOURCE_CHARS = 8000  # keep prompt within free-tier context limits (see token math in generate_quiz)

SYSTEM_PROMPT = (
    "You are a quiz-generation assistant. You output ONLY a valid JSON array, nothing else — "
    "no explanations, no markdown formatting, no code fences, no text before or after the "
    "JSON. If you include anything other than the JSON array, the output is unusable."
)


class QuizGenerationError(Exception):
    """Raised when the model's output can't be parsed into a valid quiz."""
    pass


def _extract_json(text: str):
    """Strip markdown code fences / stray prose and parse the first JSON array found."""
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        preview = text[:200].replace("\n", " ") if text else "(empty response)"
        raise QuizGenerationError(
            f"The model didn't respond with JSON — it wrote this instead: \"{preview}"
            f"{'...' if len(text) > 200 else ''}\""
        )
    json_str = text[start:end + 1]
    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        pass  # fall through to repair attempt below

    try:
        return json.loads(repair_json(json_str), strict=False)
    except (json.JSONDecodeError, Exception) as e:
        raise QuizGenerationError(
            f"Couldn't parse the model's output as JSON, even after attempting repair: {e}. "
            f"Try again, or reduce the number of questions requested."
        )


def _build_instructions(quiz_type: str, difficulty: str, num_items: int) -> str:
    if quiz_type == "MCQ":
        schema = (
            '[{"question": "...", "options": ["...", "...", "...", "..."], '
            '"correct_index": 0, "explanation": "..."}]'
        )
        task = (
            f"Generate exactly {num_items} multiple-choice questions at {difficulty} "
            "difficulty testing understanding of the document. Each question must have "
            "exactly 4 options with exactly one correct answer. correct_index is the "
            "0-based index of the correct option."
        )
    elif quiz_type == "Coding Questions":
        schema = '[{"question": "...", "hint": "...", "sample_approach": "..."}]'
        task = (
            f"Generate exactly {num_items} coding or technical exercise questions at "
            f"{difficulty} difficulty, based on the skills, tools, or concepts mentioned "
            "in the document. Include a short hint and a brief sample approach for each."
        )
    elif quiz_type == "Interview Questions":
        schema = '[{"question": "...", "what_to_cover": "..."}]'
        task = (
            f"Generate exactly {num_items} interview questions at {difficulty} difficulty "
            "based on the document. For each, briefly note what a strong answer should cover."
        )
    else:  # Flashcards
        schema = '[{"front": "...", "back": "..."}]'
        task = (
            f"Generate exactly {num_items} flashcards at {difficulty} difficulty summarizing "
            "key facts, terms, or concepts from the document. Keep the front short (a term "
            "or short question) and the back concise (the answer or definition)."
        )

    return (
        f"{task}\n\nRespond with ONLY a valid JSON array matching this schema, no prose "
        f"before or after, no markdown code fences:\n{schema}"
    )


def generate_quiz(
    document_text: str, quiz_type: str, difficulty: str, num_items: int, offset: int = 0
) -> list:
    """Generate a quiz from document_text. Returns a list of dicts whose shape depends
    on quiz_type (see _build_instructions for each schema).

    Args:
        offset: character offset into document_text to start the excerpt window
            from, for documents longer than MAX_SOURCE_CHARS. Used by the UI to
            rotate which part of a large document successive "Generate" clicks
            draw from, so batching toward 50+ items actually covers more of the
            document instead of repeatedly regenerating from the same opening slice.

    Raises:
        ValueError: invalid quiz_type/difficulty/num_items
        LLMError: no provider available or the provider call failed
        QuizGenerationError: the model's output couldn't be parsed as valid JSON
    """
    if quiz_type not in QUIZ_TYPES:
        raise ValueError(f"Unknown quiz type: {quiz_type!r}. Must be one of {QUIZ_TYPES}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty: {difficulty!r}. Must be one of {DIFFICULTIES}")
    if not (1 <= num_items <= 30):
        raise ValueError("num_items must be between 1 and 30")

    if len(document_text) > MAX_SOURCE_CHARS:
        start = offset % (len(document_text) - MAX_SOURCE_CHARS + 1)
    else:
        start = 0
    truncated = document_text[start:start + MAX_SOURCE_CHARS]
    user_prompt = (
        f"Document content:\n\n{truncated}\n\n"
        f"{_build_instructions(quiz_type, difficulty, num_items)}"
    )

    # Scale the output token budget to what was actually requested, instead of
    # a flat oversized value. A real request for just 10 items on a large
    # document previously still asked for a flat 6000 output tokens, which —
    # combined with input tokens — exceeded Groq's free-tier 8000 TPM ceiling
    # in a single request (a 413 "request too large" error, not a rate limit
    # the automatic 429 retry could help with, since the request itself was
    # simply too big regardless of recent usage).
    tokens_per_item = 180  # generous per-item estimate covering all quiz types
    base_overhead = 300
    max_tokens = min(5000, max(800, num_items * tokens_per_item + base_overhead))

    raw = generate_raw(SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens)
    try:
        items = _extract_json(raw)
    except QuizGenerationError:
        # The model occasionally responds with prose instead of JSON. Retry
        # once with a more insistent reminder before giving up.
        reinforced_prompt = (
            user_prompt + "\n\nREMINDER: your entire response must be ONLY the JSON "
            "array described above — no commentary, no explanation, before or after."
        )
        raw = generate_raw(SYSTEM_PROMPT, reinforced_prompt, max_tokens=max_tokens)
        items = _extract_json(raw)  # let this raise if it fails again

    if not isinstance(items, list) or not items:
        raise QuizGenerationError("The model didn't return a non-empty list of quiz items.")

    # Each item must be a dict (question/options/etc.) — the UI calls .get() on
    # every item, so a non-dict entry (which can happen if json-repair salvaged
    # something from badly malformed output) would otherwise crash the app.
    valid_items = [item for item in items if isinstance(item, dict)]
    if not valid_items:
        raise QuizGenerationError(
            "The model's output couldn't be parsed into usable quiz items. Try again."
        )
    return valid_items