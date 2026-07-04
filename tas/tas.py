#!/usr/bin/env python3
"""
Transcribe a Twitch-stream audio recording and summarize what was *said*.

Pipeline:
  1. ffmpeg downmixes the input to a 16 kHz mono WAV.
  2. Silero VAD (local) finds the *voiced* regions and discards purely-instrumental
     stretches. This is essential: feeding Whisper music with no speech makes it
     hallucinate looping phrases (e.g. "What do I want to do?" x37) AND drop the real
     speech buried in those windows. Only voiced audio is sent to the cloud.
  3. Voiced regions are concatenated into compressed mp3 chunks (the API caps uploads at
     25 MB) and transcribed by Azure OpenAI Whisper (verbose_json), several in parallel.
     A per-chunk mapping table converts clip-relative timestamps back to absolute stream
     time. (Chunks are independent, so concurrency is bounded only by the deployment's
     request-rate limit, not by ordering.)
  4. Azure OpenAI gpt-4.1-mini (or other LLM deployment) classifies each segment as
     TALKING or SINGING and writes a summary that covers ONLY the talking.

Outputs (in --outdir):
  <name>.segments.json  raw timestamped segments from Whisper
  <name>.marked.md      full transcript, every line tagged [TALKING] or [SINGING]
  <name>.summary.md     summary of the talking (singing excluded)

Cloud cost is pay-per-use only: Whisper bills per minute of audio, gpt-4.1-mini per
token. Nothing bills just for existing.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError, RateLimitError


_FILTER_REPORT_LOCK = Lock()
_LLM_OPTION_LOCK = Lock()
_LLM_PACE_LOCK = Lock()
_MODEL_REPORT_LOCK = Lock()
_DEFAULT_TEMPERATURE_DEPLOYMENTS: set[str] = set("gpt-5.4")
_NO_REASONING_EFFORT_DEPLOYMENTS: set[str] = set()
_RESPONSE_MODELS: dict[str, set[str]] = {"whisper": set(), "llm": set()}
_REASONING_EFFORT_CHOICES = ("none", "minimal", "low", "medium", "high")
_LLM_MIN_INTERVAL_S = 0.0
_LAST_LLM_REQUEST_AT = 0.0
_LLM_REASONING_EFFORT: str | None = "high"


# ----------------------------------------------------------------------------- helpers


def hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_dur(seconds: float) -> str:
    """Compact wall-clock duration, e.g. '4.2s', '3m05s', '1h02m03s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _response_model_name(response) -> str:
    model = getattr(response, "model", None)
    model_dump = getattr(response, "model_dump", None)
    if not model and callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                model = dumped.get("model")
        except Exception:
            model = None
    if not model and isinstance(response, dict):
        model = response.get("model")
    return str(model).strip() if model else ""


def _record_response_model(kind: str, response) -> None:
    model = _response_model_name(response)
    if not model:
        return
    with _MODEL_REPORT_LOCK:
        _RESPONSE_MODELS.setdefault(kind, set()).add(model)


def _model_display(kind: str, deployment: str) -> str:
    with _MODEL_REPORT_LOCK:
        models = sorted(_RESPONSE_MODELS.get(kind, set()))
    if not models:
        return f"{deployment} (deployment; actual model not reported)"
    model = ", ".join(models)
    if model == deployment:
        return model
    return f"{model} (deployment: {deployment})"


# The model name actually returned by the API for `kind` (first, if several), else ""
def _reported_model(kind: str) -> str:
    with _MODEL_REPORT_LOCK:
        models = sorted(_RESPONSE_MODELS.get(kind, set()))
    return models[0] if models else ""


def _reasoning_effort_display(deployment: str) -> str:
    effort = _LLM_REASONING_EFFORT or "none"
    with _LLM_OPTION_LOCK:
        omitted = deployment in _NO_REASONING_EFFORT_DEPLOYMENTS
    if _LLM_REASONING_EFFORT and omitted:
        return f"{effort} requested; omitted because deployment does not support reasoning_effort"
    return effort


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout


def probe_duration(path: Path) -> float:
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(out.strip())


SR = 16000  # Whisper's working sample rate; also what Silero VAD expects

# Azure OpenAI list prices (USD), used only for the printed cost estimate.
# Whisper transcription is billed per audio minute.
WHISPER_USD_PER_MIN = 0.006

# LLM (chat/reasoning) token prices: Global Standard list price in USD per 1M tokens, as
# (input, output). Retrieved from the Azure OpenAI pricing page on 2026-07-03; update as
# prices change. Keyed by model base name -- dated deployments (e.g.
# "gpt-4.1-mini-2025-04-14") match their base via longest-prefix lookup. Non-token-billed
# models (image, audio/realtime, embeddings) bill in different units and are omitted.
LLM_PRICING: dict[str, tuple[float, float]] = {
    # GPT-5.x family
    "gpt-chat-latest": (5.0, 30.0),
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4-pro": (30.0, 180.0),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4": (2.50, 15.0),
    "gpt-5.3-codex": (1.75, 14.0),
    "gpt-5.3-chat": (1.75, 14.0),
    "gpt-5.3": (1.75, 14.0),
    "gpt-5.2-codex": (1.75, 14.0),
    "gpt-5.2-chat": (1.75, 14.0),
    "gpt-5.2": (1.75, 14.0),
    "gpt-5.1-codex-mini": (0.25, 2.0),
    "gpt-5.1-codex-max": (1.25, 10.0),
    "gpt-5.1-codex": (1.25, 10.0),
    "gpt-5.1-chat": (1.25, 10.0),
    "gpt-5.1": (1.25, 10.0),
    "gpt-5-pro": (15.0, 120.0),
    "gpt-5-codex": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-chat": (1.25, 10.0),
    "gpt-5": (1.25, 10.0),
    # o-series reasoning
    "o3-deep-research": (10.0, 40.0),
    "o3-mini": (1.10, 4.40),
    "o3": (2.0, 8.0),
    "o4-mini": (1.10, 4.40),
    "o1-mini": (1.10, 4.40),
    "o1": (15.0, 60.0),
    # GPT-4.1 family
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.0, 8.0),
    # GPT-4o family (text)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-2024-05-13": (5.0, 15.0),
    "gpt-4o": (2.50, 10.0),
    # open-weight
    "gpt-oss-120b": (0.15, 0.60),
    # legacy
    "gpt-4-turbo": (11.0, 33.0),
    "gpt-4-32k": (60.0, 120.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo-instruct": (1.65, 2.20),
    "gpt-35-turbo-instruct": (1.65, 2.20),
    "gpt-3.5-turbo-16k": (3.0, 4.0),
    "gpt-35-turbo-16k": (3.0, 4.0),
    "gpt-3.5-turbo": (0.55, 1.65),
    "gpt-35-turbo": (0.55, 1.65),
}

# Rates assumed when the reported model isn't in LLM_PRICING (keeps a sensible estimate
# rather than $0); the stats report flags when they were assumed.
_FALLBACK_LLM_MODEL = "gpt-4.1-mini"


# find the right key for a model
def _match_pricing_key(model: str) -> str | None:
    m = (model or "").strip().lower()
    if not m:
        return None
    for key in sorted(LLM_PRICING, key=len, reverse=True):
        if m == key or m.startswith(key):
            return key
    return None


# Look up (input, output) USD-per-1M-token Global Standard rates for `model`,
# plus a short source label. Falls back to gpt-4.1-mini's rates (and says
# so) for unknown models.
def llm_token_rates(model: str) -> tuple[float, float, str]:
    key = _match_pricing_key(model)
    if key:
        rin, rout = LLM_PRICING[key]
        return rin, rout, key
    rin, rout = LLM_PRICING[_FALLBACK_LLM_MODEL]
    return rin, rout, f"assumed {_FALLBACK_LLM_MODEL} (no list price on file for '{model}')"

# ----------------------------------------------------------------------------- audio


def load_mono_16k(src: Path, workdir: Path) -> np.ndarray:
    wav = workdir / "full_16k.wav"
    print("  ffmpeg: decode -> 16 kHz mono wav ...")
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(SR),
            "-c:a",
            "pcm_s16le",
            str(wav),
        ]
    )
    samples, sr = sf.read(str(wav), dtype="float32")
    if sr != SR:
        raise RuntimeError(f"expected {SR} Hz, got {sr}")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    # The wav is intentionally left on disk so parallel VAD workers can
    # read slices of it without re-pickling the (multi-GB) sample array; the
    # caller deletes it after VAD.
    return samples


# Load the stilero VAD. onnx is preferred, but fall back if we don't have it
def _load_vad():
    from silero_vad import load_silero_vad

    try:
        return load_silero_vad(onnx=True)
    except Exception:
        return load_silero_vad(onnx=False)


def _vad_timestamps(model, samples: np.ndarray, threshold: float, max_speech_s: float):
    import torch
    from silero_vad import get_speech_timestamps

    return get_speech_timestamps(
        torch.from_numpy(samples),
        model,
        sampling_rate=SR,
        threshold=threshold,
        min_silence_duration_ms=300,  # bridge tiny pauses within an utterance
        speech_pad_ms=200,  # don't clip word edges
        max_speech_duration_s=max_speech_s,
        return_seconds=True,
    )


# Merge overlapping/adjacent (<=gap apart) regions, then split any region
# longer than max_len. With gap=0 this only removes Silero's padding-induced
# overlaps (voiced total is unchanged) and stitches regions split across
# parallel slice boundaries.
def _merge_regions(
    regions: list[tuple[float, float]], gap: float = 0.0, max_len: float | None = None
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for s, e in sorted(regions):
        if merged and s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out: list[tuple[float, float]] = []
    for s, e in merged:
        if max_len:
            while e - s > max_len:
                out.append((s, s + max_len))
                s += max_len
        out.append((s, e))
    return out


# Run VAD on one slice of the decoded wav (read directly from disk so the
# big sample array is never pickled across processes). Returns regions
# in absolute stream time.
def _vad_worker(args: tuple) -> list[tuple[float, float]]:
    wav_path, start, count, threshold, max_speech_s = args
    import soundfile as sf

    data, _ = sf.read(wav_path, start=start, frames=count, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    ts = _vad_timestamps(_load_vad(), data, threshold, max_speech_s)
    off = start / SR
    return [(off + float(t["start"]), off + float(t["end"])) for t in ts]


# Use Silero VAD to find regions containing a human voice (speech or singing).
# Purely-instrumental audio is excluded, which is what stops Whisper
# hallucinating.
#
# For long audio this splits the stream into `jobs` contiguous slices and
# runs them in parallel processes (the model is sequential, but independent
# slices are not). A 1s overlap on each side prevents speech on a slice
# boundary from being lost; regions are then merged so the result matches
# a single-process run.
def detect_voiced_regions(
    samples: np.ndarray,
    threshold: float,
    max_speech_s: float,
    jobs: int = 1,
    wav_path: Path | None = None,
) -> list[tuple[float, float]]:

    n = len(samples)
    if jobs > 1 and wav_path and n > 60 * SR:
        overlap = (
            SR  # 1 s, so a boundary-straddling utterance is fully seen by one slice
        )
        base = -(-n // jobs)
        slices = []
        for i in range(jobs):
            a = max(0, i * base - overlap)
            b = min(n, (i + 1) * base + overlap)
            if b > a:
                slices.append((str(wav_path), a, b - a, threshold, max_speech_s))
        try:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                parts = list(ex.map(_vad_worker, slices))
            return _merge_regions([r for p in parts for r in p], max_len=max_speech_s)
        except Exception as e:
            print(f"  ! parallel VAD failed ({type(e).__name__}); using single process")

    ts = _vad_timestamps(_load_vad(), samples, threshold, max_speech_s)
    return _merge_regions(
        [(float(t["start"]), float(t["end"])) for t in ts], max_len=max_speech_s
    )


# Encode a float32 mono array to a 64 kbps mp3 via ffmpeg (fed raw PCM on
# stdin). mp3 lets ~25 min of voiced audio fit under the API's 25 MB upload
# cap, so we make fewer, larger requests -- which matters because the Whisper
# deployment is rate-limited per request.
def write_mp3(samples: np.ndarray, path: Path) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "s16le", "-ar", str(SR), "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "libmp3lame", "-b:a", "64k",
            str(path),
        ],
        input=pcm,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mp3 encode failed:\n{proc.stderr.decode(errors='replace')}"
        )


# Concatenate voiced regions into <=max_chunk_s mp3 chunks. Each chunk carries
# a list of (clip_start_s, abs_start_s, dur_s) pieces for timestamp
# reconstruction.
def build_voiced_chunks(
    samples: np.ndarray,
    regions: list[tuple[float, float]],
    max_chunk_s: float,
    workdir: Path,
) -> list[tuple[Path, list]]:
    chunks: list[tuple[Path, list]] = []
    buf: list[np.ndarray] = []
    pieces: list[tuple[float, float, float]] = []
    clip_len = 0.0

    def flush() -> None:
        nonlocal buf, pieces, clip_len
        if not buf:
            return
        path = workdir / f"voiced_{len(chunks):04d}.mp3"
        write_mp3(np.concatenate(buf), path)
        chunks.append((path, pieces))
        buf, pieces, clip_len = [], [], 0.0

    for start, end in regions:
        i0, i1 = max(0, int(start * SR)), min(len(samples), int(end * SR))
        if i1 <= i0:
            continue
        dur = (i1 - i0) / SR
        if clip_len > 0 and clip_len + dur > max_chunk_s:
            flush()
        pieces.append((clip_len, start, dur))
        buf.append(samples[i0:i1])
        clip_len += dur
    flush()
    return chunks


# Map a clip-relative time back to absolute stream time via the piece table.
def map_clip_time(pieces: list[tuple[float, float, float]], t: float) -> float:
    chosen = pieces[0]
    for piece in pieces:
        if piece[0] <= t:
            chosen = piece
        else:
            break
    clip_start, abs_start, dur = chosen
    return abs_start + min(max(t - clip_start, 0.0), dur)


# Transcribe voiced chunks (up to `concurrency` at once) and map segment
# timestamps back to absolute stream time. Chunks are independent, so order
# doesn't matter; the SDK's built-in retry/backoff absorbs 429s from the
# deployment's rate limit. `prompt` (names/terms) biases Whisper toward the
# correct spellings.
def transcribe_voiced(
    client: AzureOpenAI,
    deployment: str,
    chunks: list[tuple[Path, list]],
    language: str | None,
    concurrency: int,
    prompt: str | None = None,
) -> list[dict]:
    n = len(chunks)

    def do_one(item: tuple[int, tuple[Path, list]]) -> list[dict]:
        i, (path, pieces) = item
        with open(path, "rb") as f:
            kwargs = dict(model=deployment, file=f, response_format="verbose_json")
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt
            result = client.audio.transcriptions.create(**kwargs)
        _record_response_model("whisper", result)
        out = []
        for seg in result.segments or []:
            text = (seg.text or "").strip()
            if not text:
                continue
            out.append(
                {
                    "start": round(map_clip_time(pieces, seg.start), 2),
                    "end": round(map_clip_time(pieces, seg.end), 2),
                    "text": text,
                }
            )
        print(
            f"  chunk {i + 1}/{n} done "
            f"({path.stat().st_size / 1e6:.1f} MB, {hms(sum(p[2] for p in pieces))} "
            f"voiced -> {len(out)} segments)"
        )
        return out

    segments: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        for out in ex.map(do_one, enumerate(chunks)):
            segments.extend(out)
    segments.sort(key=lambda s: (s["start"], s["end"]))
    return segments


CLASSIFY_SYSTEM = """
    You label segments from a live Twitch stream transcript. The audio mixes the streamer
    TALKING (to chat, narrating, explaining), HUMMING/SCATTING a melody ('da da dee dum',
    'dun dun da', 'la la'), and occasionally SINGING song lyrics. The streamer often hums the
    tune they are working out WHILE talking, so a single segment can contain both. Your top
    priority is to NEVER mislabel real speech as singing.

    You are given numbered segments with timestamps. Classify EACH as TALKING or SINGING:
    - If a segment contains ANY genuine conversational speech -- a remark to chat, a
    question, an answer, narration of what they're doing, a reaction, a greeting, a thank-
    you -- it is TALKING, EVEN IF it also contains humming/scat syllables. Humming syllables
    ('da da dee dum') are not real content, so humming mixed with real words is TALKING.
    - A segment is SINGING only when it has NO conversational speech: purely humming/
    scatting/vocalise, OR purely sung song lyrics, with nothing said to chat. Musical
    self-talk like note names and counting ('G sharp', 'two, three', 'verse 2') with no
    real aside also counts as SINGING.

    Judge each segment on its own. The streamer often talks in the MIDDLE of a long humming
    stretch -- those talking segments must NOT be swept up with the humming around them.
    Bias strongly toward TALKING; only mark SINGING when confident there is no real spoken
    content. Most segments in this stream are TALKING.

    Examples (humming does NOT make these SINGING):
    'do you play Pokemon Go, da dum bum, okay' -> TALKING (asks a viewer).
    'da da da, like that, having a good day Erica' -> TALKING (greets a viewer).
    'I find my phone's too heavy so I don't use it anymore' -> TALKING (no humming at all).
    'da da dee dum, ba da dee da, doo doo, two three, da da' -> SINGING (pure vocalise).

    Respond with a single JSON object, no prose:
    {"singing_indices": [list of integer indices that are SINGING; empty if none]}.
"""

SUMMARIZE_SYSTEM = """
    You summarize what was SAID during a live Twitch stream, from its transcript. The audio
    also contains the streamer humming/scatting a melody ('da da dee dum') and singing song
    lyrics -- IGNORE all of that completely. Summarize ONLY genuine spoken conversation and
    narration: topics discussed, questions answered, announcements, reactions, interactions
    with chat and viewers. Never summarize humming or sung lyrics.

    Respond with a single JSON object, no prose: {"summary": "...markdown..."}. The
    summary leads with a 1-2 sentence overview, then bullet points of the main topics/events/
    announcements in rough chronological order.
"""


def _numbered(segments: list[dict]) -> str:
    return "\n".join(
        f"[{i}] ({hms(s['start'])}-{hms(s['end'])}) {s['text']}"
        for i, s in enumerate(segments)
    )


def _add_usage(acc: dict, u) -> None:
    acc["prompt_tokens"] += u.prompt_tokens
    acc["completion_tokens"] += u.completion_tokens


def _error_body(exc: Exception) -> dict | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body
    if isinstance(body, str):
        try:
            data = json.loads(body)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            data = response.json()
            return data if isinstance(data, dict) else None
        except Exception:
            pass
    return None


def _walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _content_filter_details(exc: Exception) -> dict:
    body = _error_body(exc) or {}
    raw_error = body.get("error")
    error = raw_error if isinstance(raw_error, dict) else body
    raw_inner = error.get("innererror")
    inner = raw_inner if isinstance(raw_inner, dict) else {}

    categories: dict[str, dict] = {}
    for d in _walk_dicts(body):
        for key, value in d.items():
            if not isinstance(value, dict):
                continue
            if {"filtered", "severity", "detected"} & set(value):
                categories[key] = value

    return {
        "type": type(exc).__name__,
        "status_code": getattr(exc, "status_code", None),
        "code": getattr(exc, "code", None) or error.get("code"),
        "inner_code": inner.get("code"),
        "param": getattr(exc, "param", None) or error.get("param"),
        "message": str(error.get("message") or getattr(exc, "message", "") or exc),
        "categories": categories,
    }


def _is_content_filter_error(exc: Exception) -> bool:
    details = _content_filter_details(exc)
    haystack = (json.dumps(details, default=str) + " " + str(exc)).lower()
    return any(
        marker in haystack
        for marker in (
            "content_filter",
            "content filter",
            "content management policy",
            "responsibleaipolicyviolation",
            "responsible ai policy",
            "filtered due to",
            "policy violation",
        )
    )


def _content_filter_reason(exc: Exception) -> str:
    details = _content_filter_details(exc)
    cats = []
    for name, data in sorted(details.get("categories", {}).items()):
        severity = data.get("severity")
        filtered = data.get("filtered")
        detected = data.get("detected")
        if filtered or detected or (severity and str(severity).lower() != "safe"):
            bits = []
            if severity:
                bits.append(f"severity={severity}")
            if filtered is not None:
                bits.append(f"filtered={filtered}")
            if detected is not None:
                bits.append(f"detected={detected}")
            cats.append(f"{name}({', '.join(bits)})")
    if cats:
        return "; ".join(cats)
    return str(
        details.get("inner_code")
        or details.get("code")
        or details.get("message", "unknown reason")
    )[:240]


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    details = _content_filter_details(exc)
    haystack = (json.dumps(details, default=str) + " " + str(exc)).lower()
    return "temperature" in haystack and any(
        marker in haystack
        for marker in (
            "unsupported_value",
            "does not support",
            "only the default",
        )
    )


def _is_unsupported_reasoning_effort_error(exc: Exception) -> bool:
    details = _content_filter_details(exc)
    haystack = (json.dumps(details, default=str) + " " + str(exc)).lower()
    return ("reasoning_effort" in haystack or "reasoning effort" in haystack) and any(
        marker in haystack
        for marker in (
            "unsupported_parameter",
            "unsupported parameter",
            "unsupported_value",
            "unrecognized request argument",
            "unknown parameter",
            "does not support",
            "not supported",
        )
    )


def _rate_limit_delay(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    for key, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(key) if hasattr(headers, "get") else None
        if value is None:
            continue
        try:
            return min(120.0, max(1.0, float(value) * scale))
        except (TypeError, ValueError):
            pass
    return min(120.0, 5.0 * (2 ** max(0, attempt - 1)))


def _rate_limit_summary(exc: Exception) -> str:
    details = _content_filter_details(exc)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    interesting = (
        "retry-after",
        "retry-after-ms",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-ms-region",
    )
    found = []
    for key in interesting:
        value = headers.get(key) if hasattr(headers, "get") else None
        if value is not None:
            found.append(f"{key}={value}")
    msg = str(details.get("message") or exc).replace("\n", " ")[:180]
    return (msg + ("; " + "; ".join(found) if found else "")).strip()


def set_llm_min_interval(seconds: float) -> None:
    global _LLM_MIN_INTERVAL_S
    _LLM_MIN_INTERVAL_S = max(0.0, float(seconds))


def set_llm_reasoning_effort(effort: str | None) -> None:
    global _LLM_REASONING_EFFORT
    if effort is None or str(effort).strip().lower() == "none":
        _LLM_REASONING_EFFORT = None
    else:
        _LLM_REASONING_EFFORT = str(effort).strip().lower()


def _pace_llm_request() -> None:
    global _LAST_LLM_REQUEST_AT
    if _LLM_MIN_INTERVAL_S <= 0:
        return
    with _LLM_PACE_LOCK:
        now = time.monotonic()
        wait_s = _LAST_LLM_REQUEST_AT + _LLM_MIN_INTERVAL_S - now
        if wait_s > 0:
            time.sleep(wait_s)
            now = time.monotonic()
        _LAST_LLM_REQUEST_AT = now


def llm_completion(
    client: AzureOpenAI,
    deployment: str,
    messages: list[dict],
    response_format: dict | None = None,
):
    base_kwargs = {"model": deployment, "messages": messages}
    if response_format:
        base_kwargs["response_format"] = response_format

    rate_attempt = 0
    while True:
        kwargs = dict(base_kwargs)
        with _LLM_OPTION_LOCK:
            use_default_temperature = deployment in _DEFAULT_TEMPERATURE_DEPLOYMENTS
            use_reasoning_effort = (
                _LLM_REASONING_EFFORT is not None
                and deployment not in _NO_REASONING_EFFORT_DEPLOYMENTS
            )
        if not use_default_temperature:
            kwargs["temperature"] = 0
        if use_reasoning_effort:
            kwargs["reasoning_effort"] = _LLM_REASONING_EFFORT
        try:
            _pace_llm_request()
            response = client.chat.completions.create(**kwargs)
            _record_response_model("llm", response)
            return response
        except BadRequestError as exc:
            if "reasoning_effort" in kwargs and _is_unsupported_reasoning_effort_error(exc):
                with _LLM_OPTION_LOCK:
                    first_seen = deployment not in _NO_REASONING_EFFORT_DEPLOYMENTS
                    _NO_REASONING_EFFORT_DEPLOYMENTS.add(deployment)
                if first_seen:
                    print(
                        f"  ! {deployment} does not support reasoning_effort="
                        f"{_LLM_REASONING_EFFORT}; retrying without reasoning_effort"
                    )
                continue
            if "temperature" in kwargs and _is_unsupported_temperature_error(exc):
                with _LLM_OPTION_LOCK:
                    first_seen = deployment not in _DEFAULT_TEMPERATURE_DEPLOYMENTS
                    _DEFAULT_TEMPERATURE_DEPLOYMENTS.add(deployment)
                if first_seen:
                    print(
                        f"  ! {deployment} only supports the default temperature; "
                        "retrying without temperature=0"
                    )
                continue
            raise
        except RateLimitError as exc:
            rate_attempt += 1
            if rate_attempt > 6:
                raise
            delay = _rate_limit_delay(exc, rate_attempt)
            print(
                f"  ! {deployment} rate-limited; retrying in {fmt_dur(delay)} "
                f"(attempt {rate_attempt}/6): {_rate_limit_summary(exc)}"
            )
            time.sleep(delay)


def _segment_record(index: int | None, segment: dict) -> dict:
    out = {
        "time": hms(segment.get("start", 0.0)),
        "start": segment.get("start"),
        "end": segment.get("end"),
        "text": str(segment.get("text", "")).strip(),
    }
    if index is not None:
        out["index"] = index
    return out


def _write_content_filter_report(
    report_path: Path | None,
    phase: str,
    deployment: str,
    items: list[dict],
    exc: Exception,
    start_index: int | None = None,
) -> None:
    if not report_path or not items:
        return
    if start_index is None:
        indexed = [(None, s) for s in items]
    else:
        indexed = [(start_index + i, s) for i, s in enumerate(items)]
    if len(indexed) <= 5:
        shown = indexed
    else:
        shown = indexed[:3] + indexed[-2:]
    record = {
        "phase": phase,
        "deployment": deployment,
        "count": len(items),
        "index_start": start_index,
        "index_end": None if start_index is None else start_index + len(items) - 1,
        "time_start": hms(items[0].get("start", 0.0)),
        "time_end": hms(items[-1].get("end", items[-1].get("start", 0.0))),
        "isolated_segment": len(items) == 1,
        "reason": _content_filter_reason(exc),
        "error": _content_filter_details(exc),
        "segments": [_segment_record(i, s) for i, s in shown],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with _FILTER_REPORT_LOCK:
        with report_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# Assemble an optional CONTEXT preamble for the LLM prompts so the model
# uses the rightnames/spellings and can refer to the streamer by name.
# Returns "" if nothing was given.
def build_context_block(context: str | None, vocab: list[str]) -> str:
    parts: list[str] = []
    if context:
        parts.append(context.strip())
    if vocab:
        parts.append(
            "Names and terms that come up (use these exact spellings): "
            + ", ".join(vocab)
            + "."
        )
    if not parts:
        return ""
    return (
        "CONTEXT for this stream (use it to get names and terms right, and to refer to "
        "the streamer by name when one is given):\n" + "\n".join(parts) + "\n\n"
    )


# Read names/terms from a vocab file. Terms may be one per line and/or
# comma-separated; blank lines and lines starting with '#' are ignored.
def read_vocab_file(path: str) -> list[str]:
    terms: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.extend(t.strip() for t in line.split(",") if t.strip())
    return terms


# Combine vocab term lists, dropping case-insensitive duplicates while
# preserving first-seen order (the Whisper prompt is token-limited, so we
# don't want to waste room on repeats).
def merge_vocab(*sources: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for source in sources:
        for term in source:
            key = term.lower()
            if key not in seen:
                seen.add(key)
                out.append(term)
    return out


# Build the optional Whisper `prompt`. Whisper biases transcription toward
# words and spellings present in this prompt, so feeding it names/terms
# reduces mis-hearings.
def build_whisper_prompt(context: str | None, vocab: list[str]) -> str | None:
    bits: list[str] = []
    if context:
        bits.append(context.strip())
    if vocab:
        bits.append("Names and terms: " + ", ".join(vocab) + ".")
    prompt = " ".join(bits).strip()
    return prompt or None


# Split segments into consecutive time windows of about section_seconds each.
def _split_sections(segments: list[dict], section_seconds: float) -> list[list[dict]]:
    sections: list[list[dict]] = []
    win_start = segments[0]["start"]
    cur: list[dict] = []
    for s in segments:
        if cur and s["start"] >= win_start + section_seconds:
            sections.append(cur)
            cur, win_start = [], s["start"]
        cur.append(s)
    if cur:
        sections.append(cur)
    return sections


# Label each segment TALKING/SINGING in parallel batches. Batching keeps each
# call small enough that the model judges segments carefully -- one giant
# call over thousands of segments degrades and over-tags singing.
def classify_singing(
    client: AzureOpenAI,
    deployment: str,
    segments: list[dict],
    batch_size: int,
    concurrency: int,
    context_block: str = "",
    content_filter_report: Path | None = None,
) -> tuple[set[int], dict]:
    batches = [
        (start, segments[start : start + batch_size])
        for start in range(0, len(segments), batch_size)
    ]
    system = context_block + CLASSIFY_SYSTEM
    print(
        f"  classifying {len(segments)} segments in {len(batches)} batch(es) of "
        f"<={batch_size} via {deployment} ..."
    )

    # Classify a (sub)batch. If Azure's content filter rejects it, bisect
    # and recurseso only the offending segment(s) -- not the whole batch --
    # default to TALKING.
    def classify(start: int, batch: list[dict]) -> tuple[set[int], dict]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        try:
            resp = llm_completion(
                client,
                deployment,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Segments:\n\n{_numbered(batch)}"},
                ],
            )
        except BadRequestError as e:
            if not _is_content_filter_error(e):
                raise
            _write_content_filter_report(
                content_filter_report, "classify", deployment, batch, e, start
            )
            if len(batch) > 1:
                mid = len(batch) // 2
                ls, lu = classify(start, batch[:mid])
                rs, ru = classify(start + mid, batch[mid:])
                return ls | rs, {k: lu[k] + ru[k] for k in usage}
            print(
                f"  ! segment(s) {start}..{start + len(batch) - 1} rejected by content "
                f"filter ({_content_filter_reason(e)}); defaulting {len(batch)} to TALKING"
            )
            return set(), usage
        data = json.loads(resp.choices[0].message.content)
        local: set[int] = set()
        for idx in data.get("singing_indices", []):
            try:
                i = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(batch):
                local.add(start + i)
        _add_usage(usage, resp.usage)
        return local, usage

    singing: set[int] = set()
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        for local, u in ex.map(lambda it: classify(*it), batches):
            singing |= local
            usage["prompt_tokens"] += u["prompt_tokens"]
            usage["completion_tokens"] += u["completion_tokens"]
    return singing, usage


# Summarize the talking (the model identifies spoken content itself and
# ignores humming/singing), independent of the per-segment labels. Tries one
# pass; if that call is rejected (e.g. Azure content filter), falls back to
# map-reduce over batches so a single flagged span can't sink the whole
# summary.
def summarize(
    client: AzureOpenAI,
    deployment: str,
    segments: list[dict],
    batch_size: int = 400,
    concurrency: int = 4,
    context_block: str = "",
) -> tuple[str, dict]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    system = context_block + SUMMARIZE_SYSTEM

    def call(user: str) -> str:
        resp = llm_completion(
            client,
            deployment,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        _add_usage(usage, resp.usage)
        return json.loads(resp.choices[0].message.content).get("summary", "").strip()

    print(f"  summarizing {len(segments)} segments via {deployment} ...")
    try:
        return call(f"Transcript segments:\n\n{_numbered(segments)}"), usage
    except Exception as e:
        print(f"  ! single-pass summary failed ({type(e).__name__}); using map-reduce")

    batches = [
        segments[i : i + batch_size] for i in range(0, len(segments), batch_size)
    ]

    def part(b: list[dict]) -> str | None:
        try:
            return call(f"Transcript segments:\n\n{_numbered(b)}")
        except Exception as e:
            print(f"  ! summary batch failed ({type(e).__name__}); skipped")
            return None

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        notes = [s for s in ex.map(part, batches) if s]
    combined = "\n\n".join(notes)
    if not combined:
        return "_Summary unavailable (model calls were rejected)._", usage
    try:
        return call(
            "These are partial summaries of consecutive parts of ONE stream. Merge them into "
            "a single coherent summary (1-2 sentence overview, then chronological bullets); "
            "ignore any humming/singing:\n\n" + combined
        ), usage
    except Exception:
        return combined, usage


# Domain primer baked into the flow summary so the model interprets these streams
# correctly. Edit this if you adapt the tool to a different streamer/format.
STREAM_PRIMER = """
    BACKGROUND -- this is a music-request livestream by Dr. Jonathan Ong, who goes by 'Jon'
    (and you should refer to him as Jon). The format of the stream is: viewers bid on request
    slots; the top winners each get their requested song LEARNED live (a 'livelearn') and
    then LOOPED/produced -- Jon builds an arrangement, usually symphonic, one instrument/layer
    at a time. Raffles are sometimes run to give away free livelearns.

    The stream sometimes switches to CONCERT GRAND mode, where Jon plays a full-size concert
    grand piano with an AR overlay that animates the notes he plays. When in concert grand
    mode, viewers can shout out requests in chat, as long as they are songs that Jon knows
    (i.e. songs in the onglog sometime after 2020). Concert grand segments are normally (but
    not always) the last thing in the stream before Jon signs off. Jon talks very little
    during these segments. Most piano music played on stream is *not* from the concert grand,
    it's from the other, smaller piano (the RX2).

    Viewers support with cheers, subscriptions, gift-sub bombs, and raids, which sometimes
    leads to twitch hype trains. Jon reacts to these events on stream with thank-yous,
    shout-outs, and sometimes special alerts (in addition to the alerts that are naturally
    triggered by the various cheers and such). Completing a level 5 hype train will add time
    to the concert grand timer (concert grand will only be played if there's time left on
    the timer).

    Jon also rewards tier 3 resubs with a "resub song", where he will play a song he already
    knows, of the subscriber's choice, on the piano. These are short (a couple of minutes) and
    usually played on the RX2, and are separate from the livelearn/looping process. Most
    viewers have one specific song that they use every month for their resub songs. Resub
    songs happen in the stream at the time of the resub, and serve as a brief interrupt in
    whatever is currently happening with the stream.

    The normal structure of the stream is: Intro, a raffle (sometimes), bids for the first
    livelearn intake (after which the top 4 bids win), followed by the livelearn/looping
    process for those 4 songs. Sometimes there is a second raffle, followed by the second
    intake, usually for the top 2 bids, and then the livelearn/looping for those. If there
    is time left on the concert grand timer, and Jon has the time/energy, there is sometimes
    a concert grand segment before the stream ends.

    The last thing that happens in the stream is that there is a vote for which highlight
    videos wish to see (if there's more than one in the queue); Jon pastes the link for the
    winning highlight video into chat, and then ends the stream by raiding another streamer.

    The reader of your summary ALREADY KNOWS all of this, so do NOT restate the
    format or generic mechanics (e.g. 'he builds an arrangement by layering instruments') --
    spend your words on what was specific and interesting about THIS stream.
"""


# Parse 'HH:MM:SS' (or 'MM:SS') to seconds; unknown -> large number.
def _parse_hms(s: str) -> int:
    try:
        parts = [int(x) for x in str(s).strip().split(":")]
    except ValueError:
        return 10**9
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[-3] * 3600 + parts[-2] * 60 + parts[-1]


FLOW_SECTION_SYSTEM = """
    You are extracting structured notes from ONE portion of the stream. The transcript
    segments are each prefixed with a timestamp like '[12] (01:23:45-01:23:50) text'. The
    audio also contains humming/scatting and sung lyrics -- IGNORE those when writing notes,
    but DO use the lyrics to identify which song is being worked on. Extract, each with a
    rough timestamp drawn from the segment(s) it comes from:
    - notes: noteworthy things SAID or done -- specific topics, decisions, anecdotes,
      opinions, memorable viewer interactions. Skip filler and anything generic about how
      the stream works.
    - songs: every song being LEARNED (livelearn), LOOPED/produced, played as a resub song,
      or played on the concert grand in this portion. Give its title, the mode (one of:
      livelearn, loop, tier 3, concert grand, other), and a short note ONLY if something
      was notable about it (e.g. unusually difficult, multi-meter / odd time signature, took
      a very long time to learn, a special dedication, an unusual arrangement choice). Leave
      note empty otherwise.
    - issues: actual stream/Jon problems or concrete change requests ONLY: audio/video/gear/
      software glitches, broken equipment, login/platform failures, stream workflow problems,
      health/strain limitations that affect the stream, or explicit plans to improve the
      stream/setup. Say specifically what. DO NOT include normal music-making decisions or
      song-learning friction: picking/finding/changing an instrument or patch, transposing,
      singing down an octave, choosing a different arrangement, needing another take, fixing
      a wrong note/lyric/form/chord, not liking a song's composition, or needing to learn a
      difficult section. DO NOT include unrelated outside anecdotes/problems unless Jon or the
      stream is directly responsible for fixing them (e.g. a mall piano being in bad condition
      is not an issue unless Jon was hired to fix it).
    - events: notable community events Jon reacts to -- LARGE gift-sub bombs (include the
      count if stated), hype trains that finish LEVEL 5, raids (note who from), and raffles
      for free livelearns (note who won if stated).

    Be thorough for issues (don't miss any); be selective for notes.

    Respond with a single JSON object using these keys (use [] when a kind is absent):
    {"notes":[{"time":"HH:MM:SS","text":"..."}],"songs":[{"time":"HH:MM:SS","title":"...","mode":"...","note":"..."}],"issues":[{"time":"HH:MM:SS","text":"..."}],"events":[{"time":"HH:MM:SS","text":"..."}]}.
"""


_SOUNDEX = {
    ch: d
    for chars, d in [
        ("bfpv", "1"),
        ("cgjkqsxz", "2"),
        ("dt", "3"),
        ("l", "4"),
        ("mn", "5"),
        ("r", "6"),
    ]
    for ch in chars
}


# Vowel-insensitive consonant-code skeleton (Soundex-style). 'alinsa' and
# 'elince' both -> '452'; 'alinsavix'/'elincevix' -> '45212'. Lets phonetic
# mis-hearings of a name match even when the letters differ (Whisper renders
# 'Alinsa' as 'Elince', 'Elinsa', ...).
def _skeleton(word: str) -> str:

    out, prev = [], ""
    for ch in word.lower():
        d = _SOUNDEX.get(ch, "")
        if d:
            if d != prev:
                out.append(d)
            prev = d
        else:
            prev = ""  # vowel / h / w / y separates runs but emits no code
    return "".join(out)


# Deterministically find segment indices that plausibly name one of `names` --
# exact substring, a word fuzzy-close to the name, or a matching phonetic
# skeleton (catches mis-hearings like 'Elince'/'Elincevix'/'Elinsa' for
# 'Alinsa'). Recall-oriented and a bit loose on purpose; the LLM verifier
# afterwards prunes coincidences and non-names.
def _mention_candidates(segments: list[dict], names: list[str]) -> list[int]:
    singles = [n.lower() for n in names if " " not in n]
    skels = {n: _skeleton(n) for n in singles}
    phrases = [n.lower() for n in names if " " in n]

    def matches(w: str) -> bool:
        if len(w) < 4:
            return False
        sw = _skeleton(w)
        for n in singles:
            if n in w:
                return True
            if (
                abs(len(w) - len(n)) <= 3
                and difflib.SequenceMatcher(None, w, n).ratio() >= 0.8
            ):  # near-spelling
                return True
            # Phonetic skeleton match, but only for words that begin like the name (a vowel,
            # for a vowel-initial name) -- this keeps mis-hearings such as 'Elince'/'Elinsa'
            # while excluding the many consonant-initial words that share the skeleton
            # ('lines', 'lens', 'lance', 'loans').
            sn = skels[n]
            if len(sn) >= 3 and (w[0] in "aeiou" or w[0] == n[0]) and sw.startswith(sn):
                return True
        return False

    hits: list[int] = []
    for i, s in enumerate(segments):
        text_l = s["text"].lower()
        if any(p in text_l for p in phrases) or any(
            matches(w) for w in re.findall(r"[a-z0-9']+", text_l)
        ):
            hits.append(i)
    return hits


MENTION_VERIFY_SYSTEM = """
    You are reviewing transcript snippets produced by automatic speech-to-text, which often
    MIS-SPELLS names phonetically. Target name(s): {names}. Each snippet contains a word
    that SOUNDS like a target name. Judge by SOUND and CONTEXT, not exact spelling: a word
    that plausibly sounds like the target AND is used as a person's name, handle, or raffle
    keyword is a MATCH even if spelled differently. For example 'Elinsa', 'Elince', and
    'Elincevix' are speech-to-text renderings of 'Alinsa'/'Alinsavix' and ARE matches.

    Output one object for EVERY snippet number given (do not skip any), with a one-line
    context and a boolean "match":
    - "match": true -- the flagged word refers to a target person: addressed, thanked,
      talked about, or named as a keyword/tag (e.g. 'tag Elincevix' counts).
    - "match": false -- the flagged word is an ordinary word (e.g. 'aliens', 'alliance',
      'lines', 'analysis') or is clearly a DIFFERENT, unrelated person who merely sounds
      a bit alike (e.g. Melissa, Larissa).

    When the word is used as a name/keyword and plausibly sounds like the target, prefer true.
    Respond JSON: {"results":[{"n":<snippet number>,"match":true,"name":"...","context":"..."}]}
"""


# Find every genuine mention of the tracked names. Python finds candidate
# segments (recall), then one LLM call verifies them and writes the context
# (precision). Returns None (rather than a list) when verification could not
# be completed after retries, which signals the caller to omit the mentions
# section entirely -- the raw candidate scan is too noisy to publish unverified.
def find_mentions(
    client: AzureOpenAI, deployment: str, segments: list[dict], names: list[str]
) -> tuple[list[dict] | None, dict]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if not names:
        return [], usage
    idxs = _mention_candidates(segments, names)
    if not idxs:
        return [], usage
    blocks = []
    for n, i in enumerate(idxs):
        lo, hi = max(0, i - 1), min(len(segments), i + 2)  # +/-1 segment for context
        snippet = " ".join(segments[j]["text"] for j in range(lo, hi)).strip()
        blocks.append(f"[{n}] ({hms(segments[i]['start'])}) {snippet}")
    print(f"  checking {len(idxs)} candidate mention(s) of {', '.join(names)} ...")
    max_attempts = 3
    data = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = llm_completion(
                client,
                deployment,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": MENTION_VERIFY_SYSTEM.replace(
                            "{names}", ", ".join(names)
                        ),
                    },
                    {"role": "user", "content": "Snippets:\n\n" + "\n".join(blocks)},
                ],
            )
            _add_usage(usage, resp.usage)
            data = json.loads(resp.choices[0].message.content)
            break
        except Exception as e:
            print(
                f"  ! mention verification failed ({type(e).__name__}); "
                f"attempt {attempt}/{max_attempts}"
            )
    if data is None:
        # Verification could not be completed after retries. The raw candidate scan is far
        # too noisy to publish unverified (common words like 'almost'/'along' collide with
        # short names), so omit the mentions section entirely rather than list false hits.
        print("  ! mention verification unavailable; omitting mentions section")
        return None, usage
    # Map the model's verdicts by snippet number. A candidate is dropped ONLY if explicitly
    # marked match=false; anything omitted from the response defaults to keep (the candidate
    # scan is precise, so omission shouldn't silently lose a genuine mention).
    verdict: dict[int, tuple[bool, str, str]] = {}
    for h in data.get("results") if isinstance(data.get("results"), list) else []:
        if not isinstance(h, dict):
            continue
        try:
            raw_n = h.get("n")
            if raw_n is None:
                continue
            n = int(raw_n)
        except (TypeError, ValueError):
            continue
        verdict[n] = (
            bool(h.get("match", True)),
            str(h.get("name", "")).strip(),
            str(h.get("context", "")).strip(),
        )
    out: list[dict] = []
    for n, i in enumerate(idxs):
        match, name, ctx = verdict.get(n, (True, "", ""))
        if not match:
            continue
        out.append(
            {
                "time": hms(segments[i]["start"]),
                "name": name,
                "text": ctx or segments[i]["text"].strip(),
            }
        )
    out.sort(key=lambda b: _parse_hms(b["time"]))
    return out, usage


FLOW_REDUCE_SYSTEM = """
    You are writing the readable summary of an ENTIRE stream for someone who did not watch it,
    working from structured, time-stamped notes gathered across the stream. Produce:
    - overview: 3-5 detailed chronological paragraphs, separated by blank lines. Paragraph 1
      MUST start with the opening / first third of the stream, paragraph 2 should cover the
      middle, and the final paragraph should cover the late stream / ending. Do NOT begin the
      overview with the final act, the ending, or concert grand unless that is literally where
      the stream started. Tell the story of the whole stream: which songs were learned and
      looped and anything that made them interesting (e.g. unusually difficult, multi-meter,
      took over an hour to learn, a moving dedication, an unusual arrangement), the standout
      conversations, anecdotes and opinions, the mood and how it shifted, and the notable
      community moments. Do NOT explain the stream format or state generic mechanics -- the
      reader knows them; spend every sentence on specifics.
    - timeline: a curated, chronological list of the key moments, each with its rough
      timestamp -- the SHAPE of the stream, not a play-by-play. Aim for about {target} entries.
      ALWAYS include an entry for the start of each song's livelearn and loop, for large
      gift-sub bombs, for hype trains that reached level 5, for raids, and for free-livelearn
      raffles.
    - songs: the consolidated list across the whole stream, de-duplicated (a song learned over
      several windows appears ONCE), each with the time it started, its title, its mode
      (livelearn / loop / concert grand), and a short note only if notable.

    Ignore humming/singing.

    Respond with a single JSON object: {"overview":"...","timeline":[{"time":"HH:MM:SS","text":"..."}],"songs":[{"time":"HH:MM:SS","title":"...","mode":"...","note":"..."}]}
"""

FLOW_OVERVIEW_REWRITE_SYSTEM = """
    You are rewriting ONLY the overview of an entire stream summary. A previous overview was
    too narrow or started too late in the stream. Using the compact chronological timeline,
    song list, community events, and selected notes below, write a replacement overview.

    Requirements:
    - Write 3-5 chronological paragraphs, separated by blank lines.
    - Paragraph 1 starts with the opening / first third; paragraph 2 covers the middle; the
      final paragraph covers the late stream and ending.
    - Do NOT start with phrases like 'the final act', 'near the end', 'late in the stream',
      or with concert grand unless the stream actually began there.
    - Cover the whole stream's arc: songs/livelearns/loops, standout conversations or
      anecdotes, community moments, mood shifts, and notable choices or problems.
    - Do not explain the generic stream format.

    Respond with a single JSON object: {"overview":"..."}
"""

ISSUES_REDUCE_SYSTEM = """
    You are consolidating timestamped issue/change-request notes from a stream summary.
    Group notes that are clearly part of the same incident or recurring problem, and write
    one concise incident summary for each group. Preserve important detail: what went wrong,
    what Jon tried or changed, and how/if it resolved. Use a time range when an incident spans
    multiple notes; use a single time when it is isolated.

    First, DROP any notes that are not actual stream/Jon problems or concrete stream/setup
    change requests. Exclude normal music-making decisions or song-learning friction:
    finding/changing an instrument or patch, transposing, singing down an octave, choosing
    an arrangement, needing another take, fixing a wrong note/lyric/form/chord, disliking
    a song's composition, or working through a difficult section. Also exclude unrelated
    outside anecdotes/problems unless Jon or the stream is directly responsible for fixing
    them.

    Guidelines:
    - Combine multi-step incidents, e.g. repeated failed Twitch login/browser/dashboard
      attempts across several minutes, into one item that summarizes the progression and
      resolution.
    - Keep unrelated problems separate, even if they happen close together.
    - Keep specific technical/change-request details; do not over-compress everything into
      vague categories.
    - It is OK to return fewer items than the input notes; omit false positives entirely.
    - Return chronological items.

    Respond with a single JSON object:
    {"issues":[{"time":"HH:MM:SS or HH:MM:SS-HH:MM:SS","text":"..."}]}
"""


def _dedup(items: list[dict], keyfn) -> list[dict]:
    seen, out = set(), []
    for it in items:
        k = keyfn(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


# Readable summary that shows the flow of the stream. Per time-window it
# extracts notes, songs, technical issues / change-requests, community events,
# and any tracked-name mentions (map); then it distills a multi-paragraph
# overview + a curated timeline + a consolidated song list (reduce). Issues
# and name-mentions are reported in full, chronologically.
def summarize_flow(
    client: AzureOpenAI,
    deployment: str,
    segments: list[dict],
    section_seconds: float,
    concurrency: int,
    context_block: str = "",
    mentions: list[str] | None = None,
    content_filter_report: Path | None = None,
) -> tuple[str, dict]:
    mentions = mentions or []
    sections = _split_sections(segments, section_seconds)
    print(
        f"  summarizing {len(segments)} segments in {len(sections)} section(s), then "
        f"distilling the flow via {deployment} ..."
    )
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def call(system: str, user: str) -> dict:
        resp = llm_completion(
            client,
            deployment,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        _add_usage(usage, resp.usage)
        return json.loads(resp.choices[0].message.content)

    sec_sys = STREAM_PRIMER + context_block + FLOW_SECTION_SYSTEM

    def as_list(v) -> list:
        return v if isinstance(v, list) else []

    keys = ("notes", "songs", "issues", "events")

    # Extract a section. On a content-filter rejection, bisect and recurse
    # so only the offending segment(s) -- not the whole ~30-min window -- are
    # lost.
    def do_section(secs: list[dict]) -> dict:
        try:
            data = call(sec_sys, f"Transcript segments:\n\n{_numbered(secs)}")
            return {k: as_list(data.get(k)) for k in keys}
        except BadRequestError as e:
            if not _is_content_filter_error(e):
                raise
            _write_content_filter_report(
                content_filter_report, "flow-section", deployment, secs, e
            )
            if len(secs) > 1:
                mid = len(secs) // 2
                left, right = do_section(secs[:mid]), do_section(secs[mid:])
                return {k: left[k] + right[k] for k in keys}
            print(
                f"  ! {len(secs)} segment(s) @{hms(secs[0]['start'])} rejected by content "
                f"filter ({_content_filter_reason(e)}); skipped"
            )
            return {k: [] for k in keys}
        except Exception as e:
            print(
                f"  ! section @{hms(secs[0]['start'])} failed ({type(e).__name__}); skipped"
            )
            return {k: [] for k in keys}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        results = list(ex.map(do_section, sections))

    # Name mentions are found deterministically (scan + fuzzy) and verified by the LLM, which
    # is far more reliable than asking each section to filter to specific names.
    all_mentions, m_usage = find_mentions(client, deployment, segments, mentions)
    usage["prompt_tokens"] += m_usage["prompt_tokens"]
    usage["completion_tokens"] += m_usage["completion_tokens"]

    def gather(key: str) -> list[dict]:
        out = []
        for r in results:
            for b in r[key]:
                if isinstance(b, dict):
                    out.append(b)
                elif str(b).strip():
                    out.append({"time": "", "text": str(b).strip()})
        return out

    all_notes = gather("notes")
    all_songs = gather("songs")
    all_issues = _dedup(
        gather("issues"), lambda b: str(b.get("text", "")).strip().lower()
    )
    all_events = gather("events")
    all_issues.sort(key=lambda b: _parse_hms(b.get("time", "")))

    def summarize_issues(issues: list[dict]) -> list[dict]:
        if len(issues) < 8:
            return issues
        try:
            data = call(
                context_block + ISSUES_REDUCE_SYSTEM,
                "Issue/change-request notes to consolidate:\n\n" + render_notes(issues),
            )
            out: list[dict] = []
            for b in as_list(data.get("issues")):
                if not isinstance(b, dict):
                    continue
                text = str(b.get("text", "")).strip()
                if not text:
                    continue
                out.append({"time": str(b.get("time", "")).strip(), "text": text})
            if out:
                out.sort(key=lambda b: _parse_hms(str(b.get("time", "")).split("-")[0]))
                return out
        except Exception as e:
            print(f"  ! issue consolidation failed ({type(e).__name__}); using raw issues")
        return issues

    if not (all_notes or all_songs or all_events):
        body = (
            "_Summary unavailable (nothing spoken found, or all calls were rejected)._"
        )
        return body + _issues_md(all_issues) + _mentions_md(
            all_mentions, mentions
        ), usage

    def render_notes(notes: list[dict], label_song=False) -> str:
        lines = []
        for n in notes:
            t = str(n.get("time", "")).strip()
            if label_song:
                title = str(n.get("title", "")).strip()
                mode = str(n.get("mode", "")).strip()
                note = str(n.get("note", "")).strip()
                txt = f'"{title}" ({mode})' + (f" -- {note}" if note else "")
            else:
                txt = str(n.get("text", "")).strip()
            if not txt:
                continue
            lines.append(f"- `{t}` {txt}" if t else f"- {txt}")
        return "\n".join(lines)

    all_issues = summarize_issues(all_issues)

    total_min = (segments[-1]["end"] - segments[0]["start"]) / 60
    target = max(6, min(20, round(total_min / 25)))  # ~1 key moment per 25 min

    reduce_user = (
        "NOTES (general):\n"
        + (render_notes(all_notes) or "(none)")
        + "\n\nSONGS worked on:\n"
        + (render_notes(all_songs, label_song=True) or "(none)")
        + "\n\nCOMMUNITY EVENTS:\n"
        + (render_notes(all_events) or "(none)")
    )
    reduce_sys = (
        STREAM_PRIMER
        + context_block
        + FLOW_REDUCE_SYSTEM.replace("{target}", str(target))
    )

    overview, timeline_items, songs_items = "", [], []
    try:
        data = call(reduce_sys, reduce_user)
        overview = str(data.get("overview", "")).strip()
        for b in as_list(data.get("timeline")):
            t = str(b.get("time", "")).strip() if isinstance(b, dict) else ""
            txt = (
                str(b.get("text", "")).strip()
                if isinstance(b, dict)
                else str(b).strip()
            )
            if txt:
                timeline_items.append(f"- `{t}` {txt}" if t else f"- {txt}")
        songs_items = as_list(data.get("songs"))
    except Exception as e:
        print(f"  ! flow reduce failed ({type(e).__name__}); using raw section notes")
        timeline_items = [ln for ln in render_notes(all_notes).split("\n") if ln]
        songs_items = all_songs

    # consolidated song list (reduce output preferred; fall back to raw, de-duped by title)
    if not songs_items:
        songs_items = all_songs
    songs_items = _dedup(songs_items, lambda b: str(b.get("title", "")).strip().lower())
    songs_items.sort(key=lambda b: _parse_hms(b.get("time", "")))

    def overview_needs_rewrite(text: str) -> bool:
        if total_min < 90:
            return False
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) < 2:
            return True
        opener = paragraphs[0].lower()[:220]
        late_markers = (
            "the final act",
            "final act",
            "near the end",
            "toward the end",
            "late in the stream",
            "late in the night",
            "the closing",
            "the ending",
            "concert grand",
        )
        return any(marker in opener for marker in late_markers)

    def sample_chronological_notes(notes: list[dict], max_items: int = 60) -> list[dict]:
        cleaned = [n for n in notes if str(n.get("text", "")).strip()]
        if len(cleaned) <= max_items:
            return cleaned
        picks = sorted(
            {
                round(i * (len(cleaned) - 1) / (max_items - 1))
                for i in range(max_items)
            }
        )
        return [cleaned[i] for i in picks]

    if overview and overview_needs_rewrite(overview):
        print("  ! flow overview was too narrow; rewriting from compact timeline")
        rewrite_user = (
            "PREVIOUS OVERVIEW TO REPLACE:\n"
            + overview
            + "\n\nSONGS worked on:\n"
            + (render_notes(songs_items, label_song=True) or "(none)")
            + "\n\nCURATED TIMELINE:\n"
            + ("\n".join(timeline_items) or "(none)")
            + "\n\nCOMMUNITY EVENTS:\n"
            + (render_notes(all_events) or "(none)")
            + "\n\nSELECTED GENERAL NOTES across the stream:\n"
            + (render_notes(sample_chronological_notes(all_notes)) or "(none)")
        )
        try:
            data = call(
                STREAM_PRIMER + context_block + FLOW_OVERVIEW_REWRITE_SYSTEM,
                rewrite_user,
            )
            rewritten = str(data.get("overview", "")).strip()
            if rewritten:
                overview = rewritten
        except Exception as e:
            print(f"  ! flow overview rewrite failed ({type(e).__name__}); keeping original")

    md = overview + "\n\n" if overview else ""
    if songs_items:
        md += (
            "## Songs this stream\n\n"
            + render_notes(songs_items, label_song=True)
            + "\n\n"
        )
    if timeline_items:
        md += "## How the stream flowed\n\n" + "\n".join(timeline_items) + "\n\n"
    md += _issues_md(all_issues).lstrip("\n")
    md += _mentions_md(all_mentions, mentions)
    return md.strip() + "\n", usage


def _issues_md(issues: list[dict]) -> str:
    body = (
        "\n".join(
            f"- `{str(b.get('time', '')).strip()}` {str(b.get('text', '')).strip()}"
            for b in issues
            if str(b.get("text", "")).strip()
        )
        or "_None noted._"
    )
    return "\n\n## Issues & change requests\n\n" + body


def _mentions_md(items: list[dict] | None, mentions: list[str]) -> str:
    # items is None when verification could not be completed (after retries); omit the
    # section entirely in that case rather than publishing noisy, unverified candidates.
    if not mentions or items is None:
        return ""
    label = " / ".join(f'"{m}"' for m in mentions)
    lines = []
    for b in items:
        t = str(b.get("time", "")).strip()
        txt = str(b.get("text", "")).strip()
        nm = str(b.get("name", "")).strip()
        if not txt:
            continue
        prefix = f"**{nm}** — " if nm and len(mentions) > 1 else ""
        lines.append(f"- `{t}` {prefix}{txt}")
    body = "\n".join(lines) or "_No mentions found._"
    return f"\n\n## Mentions of {label}\n\n" + body


SECTION_SYSTEM = """
    You summarize ONE portion of a live Twitch stream from its transcript segments, which are
    each prefixed with their timestamp like '[12] (01:23:45-01:23:50) text'. The audio also
    contains humming/scatting a melody ('da da dee dum') and sung lyrics -- IGNORE all of
    that. In DETAIL, capture what was actually SAID in this portion: specific topics discussed,
    questions asked and answered, interactions and shout-outs to named viewers, decisions,
    announcements, anecdotes, opinions, and technical details. Prefer several specific,
    concrete bullets over a few vague ones; keep each bullet to one point. For EACH bullet
    include a rough timestamp marking about when it happened, taken from the timestamps of
    the segments the bullet is based on (approximate is fine -- this is just to show the flow
    of the stream, not a forensic log).

    Respond JSON: {"bullets": [{"time": "HH:MM:SS", "text": "..."}, ...]}. If only humming/
    singing occurs and nothing is said, return {"bullets": []}.
"""

OVERVIEW_SYSTEM = """
    You write a short overview (2-4 sentences) of an entire live Twitch stream, given
    detailed per-section notes from it. Capture the overall arc and main themes. Ignore any
    humming/singing.

    Respond JSON: {"summary": "...overview prose..."}.
"""


# Detailed map-reduce summary: split the stream into time windows, summarize
# each in detail (with a timestamp header), then synthesize a short overview
# on top.
def summarize_detailed(
    client: AzureOpenAI,
    deployment: str,
    segments: list[dict],
    section_seconds: float,
    concurrency: int,
    context_block: str = "",
) -> tuple[str, dict]:
    sections = _split_sections(segments, section_seconds)
    print(
        f"  summarizing {len(segments)} segments in {len(sections)} timed section(s) "
        f"via {deployment} ..."
    )

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    section_sys = context_block + SECTION_SYSTEM
    overview_sys = context_block + OVERVIEW_SYSTEM

    def call(system: str, user: str) -> dict:
        resp = llm_completion(
            client,
            deployment,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        _add_usage(usage, resp.usage)
        return json.loads(resp.choices[0].message.content)

    def do_section(secs: list[dict]) -> list[str]:
        try:
            data = call(section_sys, f"Transcript segments:\n\n{_numbered(secs)}")
            raw = data.get("bullets")
            raw = raw if isinstance(raw, list) else []
            out: list[str] = []
            for b in raw:
                if isinstance(b, dict):
                    text = str(b.get("text", "")).strip()
                    if not text:
                        continue
                    t = str(b.get("time", "")).strip()
                    out.append(f"`{t}` {text}" if t else text)
                elif str(b).strip():
                    out.append(str(b).strip())
            return out
        except Exception as e:
            print(
                f"  ! section @{hms(secs[0]['start'])} failed ({type(e).__name__}); noted"
            )
            return [f"_(this section could not be summarized: {type(e).__name__})_"]

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        section_bullets = list(ex.map(do_section, sections))

    body_parts: list[str] = []
    for secs, bullets in zip(sections, section_bullets):
        if not bullets:
            continue
        header = f"### {hms(secs[0]['start'])} – {hms(secs[-1]['end'])}"
        body_parts.append(header + "\n" + "\n".join(f"- {b}" for b in bullets))

    overview = ""
    notes = "\n".join(f"- {b}" for bl in section_bullets for b in bl)
    if notes:
        try:
            overview = (
                call(overview_sys, f"Section notes:\n\n{notes}")
                .get("summary", "")
                .strip()
            )
        except Exception:
            overview = ""

    md = (
        (overview + "\n\n" if overview else "")
        + "## Detailed timeline\n\n"
        + "\n\n".join(body_parts)
    )
    return md, usage


# ----------------------------------------------------------------------------- output


def write_outputs(
    outdir: Path, name: str, segments: list[dict], singing: set[int], summary: str
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "segments": outdir / f"{name}.segments.json",
        "marked": outdir / f"{name}.marked.md",
        "summary": outdir / f"{name}.summary.md",
    }

    paths["segments"].write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        f"# Transcript: {name}",
        "",
        "Each line is tagged `[TALKING]` or `[SINGING]`. "
        "Singing is excluded from the summary.",
        "",
    ]
    for i, s in enumerate(segments):
        tag = "SINGING" if i in singing else "TALKING"
        lines.append(f"`{hms(s['start'])}` **[{tag}]** {s['text']}")
    paths["marked"].write_text("\n\n".join(lines) + "\n", encoding="utf-8")

    talking = len(segments) - len(singing)
    header = (
        f"# Stream summary: {name}\n\n"
        f"_Based on {talking} talking segments; "
        f"{len(singing)} singing segments excluded._\n\n"
    )
    paths["summary"].write_text(header + summary + "\n", encoding="utf-8")
    return paths


def build_stats(
    name: str,
    total_s: float,
    voiced_s: float,
    billed_audio_s: float,
    n_regions: int,
    n_chunks: int,
    segments: list[dict],
    singing: set[int],
    usage: dict,
    whisper_model: str,
    llm_model: str,
    llm_reasoning_effort: str,
    llm_in_rate: float,
    llm_out_rate: float,
    rate_source: str,
    timings: dict | None = None,
    reused: bool = False,
) -> str:
    talking = len(segments) - len(singing)
    billed_min = billed_audio_s / 60
    whisper_cost = 0.0 if reused else billed_min * WHISPER_USD_PER_MIN
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    llm_cost = in_tok / 1e6 * llm_in_rate + out_tok / 1e6 * llm_out_rate
    voiced_pct = (voiced_s / total_s * 100) if total_s else 0.0

    lines = [
        f"# Stats & cost: {name}",
        "",
        "## Models",
        (
            "- Whisper transcription: reused existing segments"
            if reused
            else f"- Whisper transcription: **{whisper_model}**"
        ),
        f"- LLM classification + summary: **{llm_model}**",
        f"- LLM reasoning effort: **{llm_reasoning_effort}**",
        "",
        "## Audio",
        f"- Total duration: **{hms(total_s)}**",
    ]
    if reused:
        lines.append("- (reused existing segments; no audio sent to Whisper)")
    else:
        lines += [
            f"- Voice detected (VAD): **{hms(voiced_s)}** ({voiced_pct:.1f}% of total)",
            f"- Voiced regions: {n_regions}",
            f"- Whisper chunks uploaded: {n_chunks}",
            f"- Audio billed by Whisper: **{hms(billed_audio_s)}** ({billed_min:.1f} min)",
        ]
    lines += [
        "",
        "## Transcript",
        f"- Segments: {len(segments)} (talking {talking}, singing {len(singing)})",
        "",
        "## Cost (USD, Azure list prices)",
        "| item | usage | rate | cost |",
        "|---|---|---|---|",
    ]
    if reused:
        lines.append("| Whisper transcription | reused | - | $0.0000 |")
    else:
        lines.append(
            f"| Whisper transcription | {billed_min:.1f} min | "
            f"${WHISPER_USD_PER_MIN}/min | ${whisper_cost:.4f} |"
        )
    lines += [
        f"| LLM input | {in_tok:,} tok | ${llm_in_rate:g}/1M | "
        f"${in_tok / 1e6 * llm_in_rate:.4f} |",
        f"| LLM output | {out_tok:,} tok | ${llm_out_rate:g}/1M | "
        f"${out_tok / 1e6 * llm_out_rate:.4f} |",
        f"| **Total** | | | **${whisper_cost + llm_cost:.4f}** |",
        "",
        f"_LLM rate basis: {rate_source} (Azure Global Standard list price, per 1M tokens)._",
    ]

    if timings:
        labels = [
            ("decode", "Decode to 16 kHz mono"),
            ("vad", "Voice-activity detection"),
            ("encode", "Encode voiced mp3 chunks"),
            ("transcribe", "Whisper transcription"),
            ("classify", "Classification"),
            ("summarize", "Summarization"),
        ]
        lines += ["", "## Timing (wall clock)", "| phase | time |", "|---|---|"]
        for key, label in labels:
            if key in timings:
                lines.append(f"| {label} | {fmt_dur(timings[key])} |")
        if "total" in timings:
            lines.append(f"| **Total** | **{fmt_dur(timings['total'])}** |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="Transcribe + summarize stream audio.")
    ap.add_argument("audio", type=Path, help="input audio file (m4a, mp3, ...)")
    ap.add_argument("--outdir", type=Path, default=Path("output"))
    ap.add_argument(
        "--from-segments",
        type=Path,
        default=None,
        help="reuse an existing <name>.segments.json and skip transcription "
        "(re-runs only classification + summary; saves Whisper cost)",
    )
    ap.add_argument(
        "--force-transcribe",
        action="store_true",
        help="ignore an existing output <name>.segments.json and run Whisper again",
    )
    ap.add_argument(
        "--whisper-chunk-seconds",
        type=int,
        default=1200,
        help="max seconds of *voiced* audio per Whisper upload (mp3-compressed; "
        "fewer, larger chunks = fewer rate-limited requests)",
    )
    ap.add_argument(
        "--whisper-concurrency",
        type=int,
        default=3,
        help="parallel Whisper requests. Match the deployment's per-minute "
        "request limit (Whisper Standard caps at 3 here).",
    )
    ap.add_argument(
        "--llm-classify-batch",
        type=int,
        default=200,
        help="segments per classification call. Smaller batches classify more "
        "accurately; one huge call over thousands of segments over-tags.",
    )
    ap.add_argument(
        "--llm-concurrency",
        type=int,
        default=4,
        help="parallel LLM requests for classification and summaries. "
        "Lower this, e.g. to 1, when the LLM deployment returns 429 rate limits.",
    )
    ap.add_argument(
        "--llm-min-interval",
        type=float,
        default=0.0,
        help="minimum seconds between LLM requests across all threads. "
        "Use this to stay under token/request rate limits on small deployments.",
    )
    ap.add_argument(
        "--llm-reasoning-effort",
        choices=_REASONING_EFFORT_CHOICES,
        default=None,
        help="LLM reasoning effort to request for classification and summaries "
        "(default: LLM_REASONING_EFFORT from .env, else high; use 'none' to omit the parameter)",
    )
    ap.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD speech probability threshold (0-1). Lower catches "
        "more speech but lets in more music; higher is stricter.",
    )
    ap.add_argument(
        "--vad-jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="parallel processes for VAD (splits the audio across CPU cores; "
        f"default min(8, cpu count) = {min(8, os.cpu_count() or 1)}, "
        "capped at 8; 1 disables parallelism)",
    )
    ap.add_argument(
        "--summary-detail",
        choices=["brief", "flow", "detailed"],
        default="flow",
        help="'flow' (default) = prose overview + a short, curated timeline of the "
        "key moments with rough timestamps; 'brief' = one concise overview+"
        "bullets; 'detailed' = exhaustive per-time-section breakdown",
    )
    ap.add_argument(
        "--section-minutes",
        type=float,
        default=30.0,
        help="time-window size used when building the flow/detailed summary "
        "(default 30 min)",
    )
    ap.add_argument(
        "--context",
        default=None,
        help='free-form background on the stream (e.g. "The streamer is Jon, a '
        'musician who plays keytar"). Helps the summary use the right names '
        "and refer to the streamer by name.",
    )
    ap.add_argument(
        "--vocab",
        default=None,
        help="comma-separated names/terms that may be misheard (e.g. "
        '"Jon,Rachie,keytar,Phasmophobia"). Fed to Whisper to fix spellings '
        "and to the summary so it gets them right. Falls back to VOCAB in .env.",
    )
    ap.add_argument(
        "--vocab-file",
        default=None,
        help="path to a file of names/terms (one per line and/or comma-separated; "
        "blank lines and lines starting with '#' are ignored). Merged with --vocab. "
        "Falls back to VOCAB_FILE in .env.",
    )
    ap.add_argument(
        "--mentions",
        default=None,
        help="comma-separated names to track; when set, the flow summary adds a "
        "section listing every timestamped mention with context (default: none). "
        "Falls back to MENTIONS in .env.",
    )
    ap.add_argument(
        "--whisper-language",
        default=None,
        help="ISO-639-1 hint for Whisper (default: auto-detect)",
    )
    ap.add_argument(
        "--keep-temp", action="store_true", help="keep the intermediate audio chunks"
    )
    ap.add_argument(
        "--content-filter-report",
        type=Path,
        default=None,
        help="write JSONL diagnostics for Azure content-filter rejections "
        "(rejected ranges, isolated segments, and category/severity details)",
    )
    args = ap.parse_args()

    if args.from_segments and args.force_transcribe:
        ap.error("--force-transcribe cannot be combined with --from-segments")

    name = args.audio.stem
    auto_segments_path = args.outdir / f"{name}.segments.json"
    auto_reuse_segments = (
        args.from_segments is None
        and not args.force_transcribe
        and auto_segments_path.is_file()
    )

    if not args.from_segments and not auto_reuse_segments and not args.audio.exists():
        print(f"error: {args.audio} not found", file=sys.stderr)
        return 1

    load_dotenv()
    reasoning_effort = (
        args.llm_reasoning_effort
        or os.environ.get("LLM_REASONING_EFFORT", "high")
    ).strip().lower()
    if reasoning_effort not in _REASONING_EFFORT_CHOICES:
        ap.error(
            "LLM_REASONING_EFFORT must be one of "
            + ", ".join(_REASONING_EFFORT_CHOICES)
        )
    try:
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            max_retries=8,  # absorb 429s from the Whisper deployment's rate limit
        )
    except KeyError as e:
        print(f"error: missing env var {e} (see .env)", file=sys.stderr)
        return 1
    whisper_dep = os.environ.get("WHISPER_DEPLOYMENT", "whisper")
    llm_dep = os.environ.get("LLM_DEPLOYMENT", "gpt-4.1-mini")
    llm_concurrency = max(1, args.llm_concurrency)
    set_llm_min_interval(args.llm_min_interval)
    set_llm_reasoning_effort(reasoning_effort)
    if args.llm_min_interval > 0:
        print(f"Pacing LLM requests at >={fmt_dur(args.llm_min_interval)} apart")

    # --vocab / --vocab-file / --mentions fall back to VOCAB / VOCAB_FILE / MENTIONS in
    # .env when the flag is omitted; an explicit flag (even "") overrides the env value.
    vocab_arg = args.vocab if args.vocab is not None else os.environ.get("VOCAB")
    vocab_file = (
        args.vocab_file if args.vocab_file is not None else os.environ.get("VOCAB_FILE")
    )
    mentions_arg = (
        args.mentions if args.mentions is not None else os.environ.get("MENTIONS")
    )
    vocab = [t.strip() for t in (vocab_arg or "").split(",") if t.strip()]
    if vocab_file:
        vocab = merge_vocab(vocab, read_vocab_file(vocab_file))
    track_mentions = [t.strip() for t in (mentions_arg or "").split(",") if t.strip()]
    context_block = build_context_block(args.context, vocab)
    whisper_prompt = build_whisper_prompt(args.context, vocab)
    if context_block:
        print(
            "Using context"
            + (f" + {len(vocab)} vocab term(s)" if vocab else "")
            + " to guide transcription and summary"
        )

    if args.content_filter_report:
        args.content_filter_report.parent.mkdir(parents=True, exist_ok=True)
        args.content_filter_report.write_text("", encoding="utf-8")
        print(f"Content-filter diagnostics: {args.content_filter_report}")

    tmp_ctx = tempfile.TemporaryDirectory(prefix="tns_")
    total_s = voiced_s = billed_audio_s = 0.0
    n_regions = n_chunks = 0
    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    reused_segments = False

    if args.from_segments:
        print(f"Reusing segments from {args.from_segments} (skipping transcription)")
        segments = json.loads(args.from_segments.read_text(encoding="utf-8"))
        print(f"      -> {len(segments)} segments loaded")
        total_s = max((s["end"] for s in segments), default=0.0)
        reused_segments = True
    elif auto_reuse_segments:
        print(
            f"Found existing segments at {auto_segments_path}; reusing them "
            "(pass --force-transcribe to run Whisper again)"
        )
        segments = json.loads(auto_segments_path.read_text(encoding="utf-8"))
        print(f"      -> {len(segments)} segments loaded")
        total_s = max((s["end"] for s in segments), default=0.0)
        reused_segments = True
    else:
        print(f"Processing {args.audio} ({hms(probe_duration(args.audio))})")
        workdir = Path(
            args.keep_temp and (args.outdir / f"{name}.chunks") or tmp_ctx.name
        )
        workdir.mkdir(parents=True, exist_ok=True)

        print("[1/4] Decoding audio")
        t = time.perf_counter()
        samples = load_mono_16k(args.audio, workdir)
        timings["decode"] = time.perf_counter() - t

        vad_jobs = max(
            1, min(8, args.vad_jobs)
        )  # hard cap at 8 (diminishing returns above)
        print(
            "[2/4] Voice-activity detection (Silero VAD, ONNX"
            + (f", {vad_jobs} parallel jobs" if vad_jobs > 1 else "")
            + ")"
        )
        wav_path = workdir / "full_16k.wav"
        t = time.perf_counter()
        regions = detect_voiced_regions(
            samples,
            args.vad_threshold,
            min(args.whisper_chunk_seconds, 300),
            jobs=vad_jobs,
            wav_path=wav_path,
        )
        timings["vad"] = time.perf_counter() - t
        wav_path.unlink(missing_ok=True)  # large; only needed during VAD
        voiced = sum(e - s for s, e in regions)
        print(
            f"      -> {len(regions)} voiced regions, {hms(voiced)} of voice "
            f"out of {hms(len(samples) / SR)} total"
        )
        if not regions:
            print("No voice detected; nothing to transcribe.")
            return 0

        t = time.perf_counter()
        chunks = build_voiced_chunks(samples, regions, args.whisper_chunk_seconds, workdir)
        timings["encode"] = time.perf_counter() - t
        n_regions, n_chunks = len(regions), len(chunks)
        voiced_s, total_s = voiced, len(samples) / SR
        billed_audio_s = sum(p[2] for _, pieces in chunks for p in pieces)
        print(
            f"[3/4] Transcribing {len(chunks)} voiced chunk(s) via Whisper deployment "
            f"{whisper_dep} "
            f"({min(args.whisper_concurrency, len(chunks))} in parallel)"
        )
        t = time.perf_counter()
        segments = transcribe_voiced(
            client,
            whisper_dep,
            chunks,
            args.whisper_language,
            args.whisper_concurrency,
            prompt=whisper_prompt,
        )
        timings["transcribe"] = time.perf_counter() - t
        print(f"      -> {len(segments)} non-empty segments")
        print(f"      -> Whisper model: {_model_display('whisper', whisper_dep)}")

    if not segments:
        print("No speech detected; nothing to summarize.")
        return 0

    print(
        f"[4/4] Classifying singing vs talking + summarizing via LLM deployment {llm_dep} "
        f"(reasoning_effort={_LLM_REASONING_EFFORT or 'none'})"
    )
    t = time.perf_counter()
    singing, c_usage = classify_singing(
        client,
        llm_dep,
        segments,
        args.llm_classify_batch,
        concurrency=llm_concurrency,
        context_block=context_block,
        content_filter_report=args.content_filter_report,
    )
    timings["classify"] = time.perf_counter() - t
    t = time.perf_counter()
    if args.summary_detail == "detailed":
        summary, s_usage = summarize_detailed(
            client,
            llm_dep,
            segments,
            args.section_minutes * 60,
            concurrency=llm_concurrency,
            context_block=context_block,
        )
    elif args.summary_detail == "flow":
        summary, s_usage = summarize_flow(
            client,
            llm_dep,
            segments,
            args.section_minutes * 60,
            concurrency=llm_concurrency,
            context_block=context_block,
            mentions=track_mentions,
            content_filter_report=args.content_filter_report,
        )
    else:
        summary, s_usage = summarize(
            client,
            llm_dep,
            segments,
            concurrency=llm_concurrency,
            context_block=context_block,
        )
    timings["summarize"] = time.perf_counter() - t
    print(f"      -> LLM model: {_model_display('llm', llm_dep)}")
    print(f"      -> LLM reasoning effort: {_reasoning_effort_display(llm_dep)}")
    usage = {k: c_usage[k] + s_usage[k] for k in ("prompt_tokens", "completion_tokens")}
    timings["total"] = time.perf_counter() - t_start

    paths = write_outputs(args.outdir, name, segments, singing, summary)
    llm_reported = _reported_model("llm") or llm_dep
    llm_in_rate, llm_out_rate, rate_source = llm_token_rates(llm_reported)
    stats = build_stats(
        name,
        total_s,
        voiced_s,
        billed_audio_s,
        n_regions,
        n_chunks,
        segments,
        singing,
        usage,
        _model_display("whisper", whisper_dep),
        _model_display("llm", llm_dep),
        _reasoning_effort_display(llm_dep),
        llm_in_rate,
        llm_out_rate,
        rate_source,
        timings=timings,
        reused=reused_segments,
    )
    stats_path = args.outdir / f"{name}.stats.md"
    stats_path.write_text(stats, encoding="utf-8")
    paths["stats"] = stats_path
    if not args.keep_temp:
        tmp_ctx.cleanup()

    print("\nDone. Wrote:")
    for label, p in paths.items():
        print(f"  {label:9s} {p}")
    print("\n" + stats)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\nInterrupted; exiting.", file=sys.stderr)
        exit_code = 130  # conventional exit status for SIGINT
    raise SystemExit(exit_code)
