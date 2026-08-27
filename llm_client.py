"""
llm_client.py
Thin wrapper around free-tier LLM APIs used for answer generation, summarization,
and image/chart analysis.

Providers (in default fallback order):
    1. Groq API   — free tier, very fast, needs GROQ_API_KEY (also the only
                    provider used for vision/image analysis in this app)
    2. HF API     — Hugging Face Inference API free tier, needs HF_API_TOKEN

Which provider is used is controlled by the LLM_PROVIDER environment variable:
    LLM_PROVIDER=auto    (default) try Groq -> HF, first one available wins
    LLM_PROVIDER=groq    force Groq only
    LLM_PROVIDER=hf      force Hugging Face only

Relevant environment variables:
    GROQ_API_KEY      free key from console.groq.com
    HF_API_TOKEN      free token from huggingface.co/settings/tokens

Note on model IDs: Groq periodically deprecates and replaces hosted models.
GROQ_MODEL and GROQ_VISION_MODEL below were last verified current as of Aug 2026
against https://console.groq.com/docs/models and /docs/vision — if requests start
failing with a "model decommissioned" style error, check that page for the
current recommended replacement.
"""

import json
import os
import re
import time
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"  # only Groq model currently used that accepts image input

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# Groq's free tier has tight per-minute token limits, and vision calls in particular
# are token-heavy (the image itself counts). Automatically retry once or twice on a
# 429, using the exact wait time Groq tells us in the error message, before giving up.
GROQ_MAX_RETRIES = 2
GROQ_RETRY_JITTER = 0.5  # small buffer added on top of Groq's suggested wait

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using ONLY the "
    "provided document excerpts. If the excerpts don't contain the answer, say so clearly "
    "instead of guessing. When you use information from an excerpt, mention which source "
    "number it came from, e.g. [Source 1]. Be thorough and complete: if the user asks for "
    "a specific number of items (e.g. '20 questions', 'list all the...'), provide exactly "
    "that many, fully numbered, without stopping early or summarizing partway through. "
    "Otherwise, keep answers well-structured and free of unnecessary padding."
)


class LLMError(Exception):
    pass


def _parse_retry_after_seconds(error_text: str, default: float = 2.0) -> float:
    """Groq's 429 responses include a hint like 'Please try again in 3.375s.'
    Parse that out so we wait exactly as long as needed, not an arbitrary guess."""
    match = re.search(r"try again in ([\d.]+)s", error_text)
    if match:
        try:
            return float(match.group(1)) + GROQ_RETRY_JITTER
        except ValueError:
            pass
    return default


def _groq_post(payload: dict, api_key: str, url: str = None, stream: bool = False, timeout: int = 90):
    """POST JSON to a Groq endpoint with automatic retry on 429 (rate limit)
    responses. Defaults to the chat completions endpoint; pass `url` for other
    Groq endpoints (e.g. text-to-speech). Non-429 errors (and the final attempt
    after exhausting retries) are returned as-is for the caller to handle/raise."""
    url = url or GROQ_API_URL
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    attempt = 0
    while True:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=stream)
        if resp.status_code == 429 and attempt < GROQ_MAX_RETRIES:
            wait_seconds = _parse_retry_after_seconds(resp.text)
            time.sleep(wait_seconds)
            attempt += 1
            continue
        return resp


def _groq_post_multipart(url: str, api_key: str, files: dict, data: dict, timeout: int = 90):
    """POST multipart/form-data to a Groq endpoint (used for audio file uploads,
    e.g. speech-to-text) with the same 429 retry behavior as _groq_post."""
    headers = {"Authorization": f"Bearer {api_key}"}
    attempt = 0
    while True:
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
        if resp.status_code == 429 and attempt < GROQ_MAX_RETRIES:
            wait_seconds = _parse_retry_after_seconds(resp.text)
            time.sleep(wait_seconds)
            attempt += 1
            continue
        return resp


def _build_context(chunks) -> str:
    """Format retrieved chunks into a numbered context block the model can cite."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Source {i}] ({c.filename}, page {c.page}):\n{c.text}")
    return "\n\n".join(parts)


def _call_groq(question: str, context: str, api_key: str) -> str:
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}"},
        ],
        "temperature": 0.2,
        "max_tokens": 3000,
    }
    resp = _groq_post(payload, api_key, timeout=60)
    if resp.status_code != 200:
        raise LLMError(f"Groq API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_hf(question: str, context: str, api_key: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = (
        f"<s>[INST] {SYSTEM_PROMPT}\n\nDocument excerpts:\n\n{context}\n\n"
        f"Question: {question} [/INST]"
    )
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 1500, "temperature": 0.2}}
    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise LLMError(f"HF API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        text = data[0]["generated_text"]
        return text.split("[/INST]")[-1].strip()
    raise LLMError(f"Unexpected HF response format: {data}")


def _call_groq_raw(system_prompt: str, user_prompt: str, api_key: str, max_tokens: int = 3000) -> str:
    """Like _call_groq but takes a fully custom system/user prompt instead of the
    citation-style Q&A format. Used by features that need structured output
    (e.g. JSON) rather than a grounded, cited answer."""
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    resp = _groq_post(payload, api_key, timeout=90)
    if resp.status_code != 200:
        raise LLMError(f"Groq API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_hf_raw(system_prompt: str, user_prompt: str, api_key: str, max_new_tokens: int = 1500) -> str:
    """HF equivalent of _call_groq_raw."""
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.3}}
    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        raise LLMError(f"HF API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        text = data[0]["generated_text"]
        return text.split("[/INST]")[-1].strip()
    raise LLMError(f"Unexpected HF response format: {data}")


def generate_raw(system_prompt: str, user_prompt: str, max_tokens: int = 3000) -> str:
    """Generic completion call with a fully custom system/user prompt (no citation
    formatting, no retrieved-chunks assumption). Same Groq -> HF provider fallback
    as generate_answer. Used for structured-output features like quiz generation.
    """
    provider = os.environ.get("LLM_PROVIDER", "auto").lower()
    groq_key = os.environ.get("GROQ_API_KEY")
    hf_key = os.environ.get("HF_API_TOKEN")

    if provider == "groq":
        if not groq_key:
            raise LLMError("LLM_PROVIDER=groq but GROQ_API_KEY is not set.")
        return _call_groq_raw(system_prompt, user_prompt, groq_key, max_tokens=max_tokens)

    if provider == "hf":
        if not hf_key:
            raise LLMError("LLM_PROVIDER=hf but HF_API_TOKEN is not set.")
        return _call_hf_raw(system_prompt, user_prompt, hf_key, max_new_tokens=min(max_tokens, 1500))

    errors = []
    if groq_key:
        try:
            return _call_groq_raw(system_prompt, user_prompt, groq_key, max_tokens=max_tokens)
        except Exception as e:
            errors.append(f"Groq: {e}")
    if hf_key:
        try:
            return _call_hf_raw(system_prompt, user_prompt, hf_key, max_new_tokens=min(max_tokens, 1500))
        except Exception as e:
            errors.append(f"HF: {e}")

    if not groq_key and not hf_key and not errors:
        raise LLMError(
            "No LLM provider available. Set GROQ_API_KEY (free at console.groq.com) "
            "or HF_API_TOKEN (free at huggingface.co/settings/tokens)."
        )
    raise LLMError("All configured LLM providers failed: " + " | ".join(errors))


def _call_groq_stream(question: str, context: str, api_key: str):
    """Stream tokens from Groq as they're generated (Server-Sent Events)."""
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}"},
        ],
        "temperature": 0.2,
        "max_tokens": 3000,
        "stream": True,
    }
    resp = _groq_post(payload, api_key, stream=True, timeout=60)
    if resp.status_code != 200:
        raise LLMError(f"Groq API error {resp.status_code}: {resp.text[:300]}")

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            break
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        delta = event.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if content:
            yield content


def generate_answer_stream(question: str, chunks):
    """Generator version of generate_answer — yields answer text piece by piece as it's
    generated, so the UI can render it word-by-word instead of waiting for the full answer.

    Provider selection mirrors generate_answer's LLM_PROVIDER logic.

    Hugging Face's Inference API doesn't support reliable token streaming for arbitrary
    models, so that path yields the full answer as a single chunk instead of failing.
    """
    if not chunks:
        yield "I couldn't find any relevant content in the uploaded documents to answer that."
        return

    context = _build_context(chunks)
    provider = os.environ.get("LLM_PROVIDER", "auto").lower()

    groq_key = os.environ.get("GROQ_API_KEY")
    hf_key = os.environ.get("HF_API_TOKEN")

    if provider == "groq":
        if not groq_key:
            raise LLMError("LLM_PROVIDER=groq but GROQ_API_KEY is not set.")
        yield from _call_groq_stream(question, context, groq_key)
        return

    if provider == "hf":
        if not hf_key:
            raise LLMError("LLM_PROVIDER=hf but HF_API_TOKEN is not set.")
        yield _call_hf(question, context, hf_key)
        return

    # auto mode: pick the first available provider, same priority as generate_answer
    if groq_key:
        yield from _call_groq_stream(question, context, groq_key)
        return
    if hf_key:
        yield _call_hf(question, context, hf_key)
        return

    raise LLMError(
        "No LLM provider available. Set GROQ_API_KEY (free at console.groq.com) "
        "or HF_API_TOKEN (free at huggingface.co/settings/tokens)."
    )


def generate_answer(question: str, chunks) -> str:
    """Generate an answer grounded in retrieved chunks.

    Provider selection follows LLM_PROVIDER (see module docstring). In "auto" mode
    (the default) it tries Groq, then Hugging Face, using whichever is configured first.
    """
    if not chunks:
        return "I couldn't find any relevant content in the uploaded documents to answer that."

    context = _build_context(chunks)
    provider = os.environ.get("LLM_PROVIDER", "auto").lower()

    groq_key = os.environ.get("GROQ_API_KEY")
    hf_key = os.environ.get("HF_API_TOKEN")

    if provider == "groq":
        if not groq_key:
            raise LLMError("LLM_PROVIDER=groq but GROQ_API_KEY is not set.")
        return _call_groq(question, context, groq_key)

    if provider == "hf":
        if not hf_key:
            raise LLMError("LLM_PROVIDER=hf but HF_API_TOKEN is not set.")
        return _call_hf(question, context, hf_key)

    # provider == "auto": try each in order, collecting errors as we go
    errors = []
    if groq_key:
        try:
            return _call_groq(question, context, groq_key)
        except Exception as e:
            errors.append(f"Groq: {e}")
    if hf_key:
        try:
            return _call_hf(question, context, hf_key)
        except Exception as e:
            errors.append(f"HF: {e}")

    if not groq_key and not hf_key and not errors:
        raise LLMError(
            "No LLM provider available. Set GROQ_API_KEY (free at console.groq.com) "
            "or HF_API_TOKEN (free at huggingface.co/settings/tokens)."
        )
    raise LLMError("All configured LLM providers failed: " + " | ".join(errors))


def summarize_document(full_text: str) -> str:
    """Summarize an entire document (used for the 'AI summarizes research papers' feature).
    Truncates very long documents to stay within context limits of free-tier models."""
    max_chars = 12000
    truncated = full_text[:max_chars]
    question = (
        "Summarize this research paper in 5-8 bullet points, covering: the research "
        "question, methodology, key findings, and conclusions."
    )
    class _FakeChunk:
        def __init__(self, text):
            self.text = text
            self.filename = "document"
            self.page = 1
    return generate_answer(question, [_FakeChunk(truncated)])


def _call_groq_vision(image_b64: str, mime_type: str, prompt: str, api_key: str, max_tokens: int = 1500) -> str:
    """Call Groq's vision-capable model with a base64-encoded image plus a text prompt."""
    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                ],
            }
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    resp = _groq_post(payload, api_key, timeout=90)
    if resp.status_code != 200:
        raise LLMError(f"Groq Vision API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def analyze_image(image_b64: str, mime_type: str = "image/png", prompt: str = None, max_tokens: int = 1500) -> str:
    """Analyze an image (e.g. a rendered PDF page or an extracted figure) using a
    vision-capable model. Only Groq is used here — Hugging Face's free Inference
    API doesn't reliably support vision across arbitrary models, so there's no
    fallback provider for this specific feature.

    Args:
        image_b64: base64-encoded image bytes (no data: prefix).
        mime_type: e.g. "image/png" or "image/jpeg".
        prompt: what to ask about the image. Defaults to a general description
            + chart/table data extraction prompt if not provided.
        max_tokens: response length cap.

    Raises:
        LLMError: GROQ_API_KEY isn't set, or the API call failed.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise LLMError(
            "Image analysis requires a Groq API key (GROQ_API_KEY) — Groq is currently "
            "the only configured provider with vision support in this app."
        )
    default_prompt = (
        "Describe what's in this image in detail. If it contains a chart, graph, or "
        "table, extract and explain the key data, trends, axis labels, and any "
        "numeric values visible. If it's a diagram, explain what it illustrates. "
        "If it's a photo or figure, describe its content and relevance."
    )
    return _call_groq_vision(image_b64, mime_type, prompt or default_prompt, groq_key, max_tokens=max_tokens)