"""
translator.py
Translates text (pasted directly, or an uploaded document's content) into a
target language using the existing LLM providers (Groq/HF text generation —
no separate translation API needed).
"""

from llm_client import generate_raw

MAX_CHARS = 8000  # keep translation requests within free-tier context limits

SYSTEM_PROMPT = (
    "You are a precise, fluent translator. Translate the user's text into the "
    "requested target language. Preserve meaning, tone, and formatting (line "
    "breaks, lists) as closely as possible. Output ONLY the translated text — "
    "no commentary, no explanations, no notes about the translation."
)


def translate_text(text: str, target_language: str) -> str:
    """Translate text into target_language.

    Raises:
        ValueError: empty text or empty target language
        LLMError: no provider available or the provider call failed
    """
    if not text or not text.strip():
        raise ValueError("There's no text to translate.")
    if not target_language or not target_language.strip():
        raise ValueError("A target language is required.")

    truncated = text[:MAX_CHARS]
    truncated_note = ""
    if len(text) > MAX_CHARS:
        truncated_note = f"\n\n*(Note: input was truncated to the first {MAX_CHARS} characters.)*"

    user_prompt = f"Translate the following text into {target_language.strip()}:\n\n{truncated}"
    result = generate_raw(SYSTEM_PROMPT, user_prompt, max_tokens=3000)
    return result + truncated_note