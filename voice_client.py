"""
voice_client.py
Speech-to-text and text-to-speech via Groq's audio endpoints.

Speech-to-text: whisper-large-v3-turbo, free tier (2,000 requests/day, no
credit card required, per console.groq.com/docs/speech-to-text as of Aug 2026).

Text-to-speech: canopylabs/orpheus-v1-english (Orpheus). IMPORTANT: unlike the
text/vision/STT endpoints this app already uses, Groq's TTS pricing page
(console.groq.com/docs/text-to-speech/orpheus) does not advertise a free-tier
allowance the way the others do — treat this as a paid-per-character feature
($22 / 1M characters as of Aug 2026) and surface that to the user rather than
assuming it's free. It also has a hard 200-character limit per request, so
longer text is chunked into multiple calls and the resulting WAV clips are
concatenated into one playable file.
"""

import io
import os
import re
import wave

from llm_client import _groq_post, _groq_post_multipart, LLMError

STT_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
TTS_API_URL = "https://api.groq.com/openai/v1/audio/speech"

STT_MODEL = "whisper-large-v3-turbo"
TTS_MODEL = "canopylabs/orpheus-v1-english"

TTS_MAX_CHARS_PER_CALL = 200  # hard limit imposed by the Orpheus API
TTS_DEFAULT_MAX_TOTAL_CHARS = 600  # cap total chars spoken per "read aloud" (cost/rate-limit control)
TTS_DEFAULT_VOICE = "troy"
TTS_VALID_VOICES = {"autumn", "diana", "hannah", "austin", "daniel", "troy"}


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Transcribe recorded speech to text using Groq's Whisper API.

    Raises:
        LLMError: no Groq API key configured, or the transcription call failed
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise LLMError(
            "Voice input requires a Groq API key (GROQ_API_KEY) — Groq is currently "
            "the only configured provider with speech-to-text support in this app."
        )
    files = {"file": (filename, audio_bytes)}
    data = {"model": STT_MODEL, "response_format": "json"}
    resp = _groq_post_multipart(STT_API_URL, groq_key, files=files, data=data, timeout=60)
    if resp.status_code != 200:
        raise LLMError(f"Groq transcription error {resp.status_code}: {resp.text[:300]}")
    text = resp.json().get("text", "").strip()
    if not text:
        raise LLMError("No speech was detected in the recording. Try again and speak clearly.")
    return text


def _split_for_tts(text: str, max_chars: int = TTS_MAX_CHARS_PER_CALL) -> list:
    """Split text into chunks that each fit within Orpheus's per-request character
    limit, breaking on sentence boundaries where possible so each chunk sounds
    natural rather than cutting off mid-sentence."""
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i:i + max_chars])
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks


def _concatenate_wavs(wav_byte_chunks: list) -> bytes:
    """Concatenate multiple WAV clips (same format) into a single playable WAV."""
    if len(wav_byte_chunks) == 1:
        return wav_byte_chunks[0]

    with wave.open(io.BytesIO(wav_byte_chunks[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]

    for wav_bytes in wav_byte_chunks[1:]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            frames.append(w.readframes(w.getnframes()))

    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as out:
        out.setparams(params)
        for f in frames:
            out.writeframes(f)
    return out_buf.getvalue()


def synthesize_speech(
    text: str,
    voice: str = TTS_DEFAULT_VOICE,
    max_total_chars: int = TTS_DEFAULT_MAX_TOTAL_CHARS,
) -> bytes:
    """Convert text to speech using Groq's Orpheus TTS model. Text longer than
    max_total_chars is truncated first (to control cost and API call count —
    see module docstring re: TTS not being confirmed free-tier), then split
    into <=200-char chunks (Orpheus's hard limit) and the resulting audio
    clips are concatenated into one WAV file.

    Raises:
        ValueError: empty text
        LLMError: no Groq API key configured, or a synthesis call failed
    """
    if not text or not text.strip():
        raise ValueError("There's no text to speak.")

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise LLMError(
            "Voice output requires a Groq API key (GROQ_API_KEY) — Groq is currently "
            "the only configured provider with text-to-speech support in this app."
        )

    if voice not in TTS_VALID_VOICES:
        voice = TTS_DEFAULT_VOICE

    truncated = text.strip()[:max_total_chars]
    chunks = _split_for_tts(truncated)
    if not chunks:
        raise ValueError("There's no text to speak.")

    wav_parts = []
    for chunk in chunks:
        payload = {"model": TTS_MODEL, "input": chunk, "voice": voice, "response_format": "wav"}
        resp = _groq_post(payload, groq_key, url=TTS_API_URL, timeout=60)
        if resp.status_code != 200:
            raise LLMError(f"Groq TTS error {resp.status_code}: {resp.text[:300]}")
        wav_parts.append(resp.content)

    return _concatenate_wavs(wav_parts)