"""Gemini 3.8 TTS Gradio UI — single text and batch files."""

from __future__ import annotations

import base64
import io
import os
import random
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from text_prep import (
    parse_dialogue_turns,
    prepare_text,
    split_text_on_pauses,
    style_for_speed,
)

load_dotenv()

try:
    from google import genai
except ImportError as exc:
    raise ImportError(
        "google-genai is required. Install with: uv pip install -r requirements.txt"
    ) from exc


SAMPLE_RATE = 24000
CHUNK_GAP_SEC = 0.15
EXPORT_BITRATE = "32k"
COMPRESSED_FORMATS = {"mp3", "aac"}
TEXT_FILE_SUFFIXES = {".txt", ".md", ".text", ".markdown"}
MAX_RETRIES = 4

FLASH_MODEL = "gemini-3.8-flash-tts"
LITE_MODEL = "gemini-3.8-flash-lite-tts"

MODEL_PRESETS = {
    FLASH_MODEL: {
        "label": "Gemini 3.8 Flash TTS",
        "subtitle": "Studio-grade · 130 languages · acting & dialects",
        "info": (
            "Flagship creative model: audiobooks, multi-speaker dialogue, "
            "regional accents, and vocal-burst acting."
        ),
    },
    LITE_MODEL: {
        "label": "Gemini 3.8 Flash-Lite TTS",
        "subtitle": "Fast · cost-efficient · 101 languages",
        "info": (
            "High-throughput workhorse: batch production, read-aloud, "
            "and everyday single-speaker generation."
        ),
    },
}
DEFAULT_MODEL = FLASH_MODEL
MODEL_DROPDOWN = [(spec["label"], key) for key, spec in MODEL_PRESETS.items()]

# Prebuilt studio voices from the Gemini TTS Voice options table.
PREBUILT_VOICES = [
    ("Zephyr — Bright", "Zephyr"),
    ("Puck — Upbeat", "Puck"),
    ("Charon — Informative", "Charon"),
    ("Kore — Firm", "Kore"),
    ("Fenrir — Excitable", "Fenrir"),
    ("Leda — Youthful", "Leda"),
    ("Orus — Firm", "Orus"),
    ("Aoede — Breezy", "Aoede"),
    ("Callirrhoe — Easy-going", "Callirrhoe"),
    ("Autonoe — Bright", "Autonoe"),
    ("Enceladus — Breathy", "Enceladus"),
    ("Iapetus — Clear", "Iapetus"),
    ("Umbriel — Easy-going", "Umbriel"),
    ("Algieba — Smooth", "Algieba"),
    ("Despina — Smooth", "Despina"),
    ("Erinome — Clear", "Erinome"),
    ("Algenib — Gravelly", "Algenib"),
    ("Rasalgethi — Informative", "Rasalgethi"),
    ("Laomedeia — Upbeat", "Laomedeia"),
    ("Achernar — Soft", "Achernar"),
    ("Alnilam — Firm", "Alnilam"),
    ("Schedar — Even", "Schedar"),
    ("Gacrux — Mature", "Gacrux"),
    ("Pulcherrima — Forward", "Pulcherrima"),
    ("Achird — Friendly", "Achird"),
    ("Zubenelgenubi — Casual", "Zubenelgenubi"),
    ("Vindemiatrix — Gentle", "Vindemiatrix"),
    ("Sadachbia — Lively", "Sadachbia"),
    ("Sadaltager — Knowledgeable", "Sadaltager"),
    ("Sulafat — Warm", "Sulafat"),
]
DEFAULT_VOICE = "Kore"

RANDOM_QUOTES = [
    "Hello, this is Gemini text to speech. I can read long documents, stories, and notes out loud.",
    "The weather is clear this morning, and the river looks almost still from here.",
    "Please leave a message after the tone. I will call you back before the end of the day.",
    "Chapter one. The ship left harbor at dawn, and nobody on deck spoke for a long while.",
    "If you can hear this, the microphone is working, and the voice settings are ready.",
]


def log_progress(message, level="INFO"):
    colors = {
        "INFO": "\033[94m",
        "DEBUG": "\033[92m",
        "WARN": "\033[93m",
        "ERROR": "\033[91m",
        "ENDC": "\033[0m",
    }
    color = colors.get(level, colors["INFO"])
    timestamp = time.strftime("%H:%M:%S")
    print(f"{color}[{timestamp} - {level}] {message}{colors['ENDC']}")


_client_lock = threading.Lock()
_client_cache: dict[str, object] = {}


def resolve_api_key(ui_key: str | None = None) -> str:
    key = (ui_key or "").strip()
    if key:
        return key
    return (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("GOOGLE_GENAI_API_KEY", "").strip()
    )


def get_client(api_key: str):
    if not api_key:
        raise RuntimeError(
            "No Gemini API key. Set GEMINI_API_KEY in the environment, a .env file, "
            "or the API key field in the UI. Get a key from Google AI Studio."
        )
    with _client_lock:
        client = _client_cache.get(api_key)
        if client is None:
            client = genai.Client(api_key=api_key)
            _client_cache[api_key] = client
        return client


def resolve_voice(voice: str, custom_voice: str | None) -> str:
    custom = (custom_voice or "").strip()
    if custom:
        return custom
    return (voice or DEFAULT_VOICE).strip() or DEFAULT_VOICE


def pcm_to_float32(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros(0, dtype=np.float32)
    if data.startswith(b"RIFF"):
        with wave.open(io.BytesIO(data), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            width = wf.getsampwidth()
        if width == 2:
            audio = np.frombuffer(frames, dtype=np.int16)
        elif width == 1:
            audio = np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128
            audio = audio * 256
        else:
            audio = np.frombuffer(frames, dtype=np.int16)
        return audio.astype(np.float32) / 32768.0
    audio = np.frombuffer(data, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


def silence_array(seconds: float) -> np.ndarray:
    n = max(0, int(SAMPLE_RATE * float(seconds)))
    return np.zeros(n, dtype=np.float32)


def extract_audio_bytes(interaction) -> bytes:
    output_audio = getattr(interaction, "output_audio", None)
    if output_audio is not None:
        data = getattr(output_audio, "data", None)
        if data:
            if isinstance(data, str):
                return base64.b64decode(data)
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
    steps = getattr(interaction, "steps", None) or []
    for step in steps:
        contents = getattr(step, "content", None) or []
        for item in contents:
            if getattr(item, "type", None) == "audio" and getattr(item, "data", None):
                data = item.data
                return base64.b64decode(data) if isinstance(data, str) else bytes(data)
    raise RuntimeError("Gemini returned no audio data.")


def _is_retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    retry_tokens = ("429", "500", "502", "503", "504", "unavailable", "resource exhausted", "deadline")
    return any(token in text for token in retry_tokens)


def synthesize_request(client, model: str, payload: dict) -> np.ndarray:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            interaction = client.interactions.create(**payload)
            audio_bytes = extract_audio_bytes(interaction)
            return pcm_to_float32(audio_bytes)
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES or not _is_retryable(exc):
                raise
            delay = min(16.0, 1.5 ** attempt) + random.random() * 0.3
            log_progress(
                f"Retry {attempt}/{MAX_RETRIES} after error: {exc} (sleep {delay:.1f}s)",
                "WARN",
            )
            time.sleep(delay)
    raise last_error  # pragma: no cover


def single_speaker_payload(model: str, text: str, voice: str, style: str) -> dict:
    annotations = [{"type": "speech_metadata"}]
    if style:
        annotations[0]["style"] = style
    return {
        "model": model,
        "input": [
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "annotations": annotations,
                    }
                ],
            }
        ],
        "response_format": {
            "type": "audio",
            "mime_type": "audio/l16",
            "sample_rate": SAMPLE_RATE,
        },
        "generation_config": {
            "speech_config": [{"voice": voice}],
        },
    }


def multi_speaker_payload(
    model: str,
    turns: list[tuple[str, str, str]],
    speakers: list[tuple[str, str]],
) -> dict:
    content = []
    for speaker, text, style in turns:
        annotations = [{"type": "speech_metadata", "speaker": speaker}]
        if style:
            annotations[0]["style"] = style
        content.append({"type": "text", "text": text, "annotations": annotations})
    return {
        "model": model,
        "input": [{"type": "user_input", "content": content}],
        "response_format": {
            "type": "audio",
            "mime_type": "audio/l16",
            "sample_rate": SAMPLE_RATE,
        },
        "generation_config": {
            "speech_config": {
                "speakers": [
                    {"speaker": name, "voice": voice} for name, voice in speakers
                ]
            }
        },
    }


def process_chunk(index: int, payload: dict, client, preview: str):
    try:
        audio = synthesize_request(client, payload["model"], payload)
        return index, audio, preview
    except Exception as exc:
        log_progress(f"Audio generation failed for chunk {index + 1}: {exc}", "ERROR")
        return index, None, f"[error chunk {index + 1}] {exc}"


def generate_chunk_audio(jobs: list[tuple[dict, str]], client, parallel_chunks: int):
    if not jobs:
        return [], []
    results = [None] * len(jobs)
    workers = max(1, int(parallel_chunks) if parallel_chunks else 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        log_progress(
            f"Submitting {len(jobs)} chunk(s) to thread pool ({workers} workers)...",
            "DEBUG",
        )
        futures = [
            executor.submit(process_chunk, i, payload, client, preview)
            for i, (payload, preview) in enumerate(jobs)
        ]
        completed = 0
        for future in as_completed(futures):
            completed += 1
            index, audio, note = future.result()
            if audio is not None:
                results[index] = (audio, note)
            else:
                log_progress(f"Chunk {index + 1} produced no audio.", "WARN")
            log_progress(f"Completed chunk processing: {completed}/{len(jobs)}.", "INFO")

    for i, res in enumerate(results):
        if res is not None:
            continue
        log_progress(f"Retrying chunk {i + 1} serially...", "WARN")
        payload, preview = jobs[i]
        _, audio, note = process_chunk(i, payload, client, preview)
        if audio is not None:
            results[i] = (audio, note)
        else:
            log_progress(f"Chunk {i + 1} skipped after retry.", "ERROR")

    missing = [i + 1 for i, res in enumerate(results) if res is None]
    if missing:
        log_progress(f"Output is incomplete; skipped chunk(s) {missing}.", "ERROR")
    all_audio = [res[0] for res in results if res]
    all_notes = [res[1] for res in results if res]
    return all_audio, all_notes


def _concat_audio(pieces: list[np.ndarray], gap_sec: float = 0.0) -> np.ndarray | None:
    if not pieces:
        return None
    gap = silence_array(gap_sec) if gap_sec > 0 else None
    out = []
    for i, audio in enumerate(pieces):
        out.append(np.asarray(audio, dtype=np.float32))
        if gap is not None and i < len(pieces) - 1:
            out.append(gap)
    return np.concatenate(out)


def build_jobs(
    text: str,
    model: str,
    voice: str,
    style: str,
    multi_speaker: bool,
    speaker1_name: str,
    speaker1_voice: str,
    speaker2_name: str,
    speaker2_voice: str,
    clean_lowercase: bool,
    clean_whitespace: bool,
    clean_references: bool,
    clean_initials: bool,
) -> list[tuple[dict, str]]:
    jobs: list[tuple[dict, str]] = []
    dialogue = parse_dialogue_turns(text) if multi_speaker else None
    if multi_speaker and dialogue:
        name1 = (speaker1_name or "").strip() or "Speaker 1"
        name2 = (speaker2_name or "").strip() or "Speaker 2"
        voice1 = speaker1_voice or voice
        voice2 = speaker2_voice or voice
        configured = {name1.lower(), name2.lower()}
        unknown = {name for name, _ in dialogue if name.lower() not in configured}
        if unknown:
            raise RuntimeError(
                "Multi-speaker transcript uses names that are not configured: "
                + ", ".join(sorted(unknown))
                + f". Set Speaker 1 / Speaker 2 to match ({name1}, {name2})."
            )
        grouped: list[list[tuple[str, str]]] = []
        current: list[tuple[str, str]] = []
        current_len = 0
        for speaker, utterance in dialogue:
            if current_len + len(utterance) > 2400 and current:
                grouped.append(current)
                current = []
                current_len = 0
            current.append((speaker, utterance))
            current_len += len(utterance)
        if current:
            grouped.append(current)
        speakers = [(name1, voice1), (name2, voice2)]
        for group in grouped:
            turns = [(speaker, utterance, style) for speaker, utterance in group]
            preview = " / ".join(f"{s}: {u[:40]}" for s, u in group)
            jobs.append((multi_speaker_payload(model, turns, speakers), preview))
        return jobs

    chunks = prepare_text(
        text,
        clean_lowercase,
        clean_whitespace,
        clean_references,
        clean_initials,
    )
    for chunk in chunks:
        preview = chunk if len(chunk) <= 80 else chunk[:77] + "..."
        jobs.append((single_speaker_payload(model, chunk, voice, style), preview))
    return jobs


def generate_first(
    text: str,
    api_key: str,
    model: str,
    voice: str,
    custom_voice: str,
    speed: float,
    style: str,
    multi_speaker: bool,
    speaker1_name: str,
    speaker1_voice: str,
    speaker2_name: str,
    speaker2_voice: str,
    clean_lowercase: bool,
    clean_whitespace: bool,
    clean_references: bool,
    clean_initials: bool,
    parallel_chunks: int,
):
    log_progress("Generation started.", "INFO")
    text = (text or "").strip()
    if not text:
        log_progress("Input text is empty. Aborting.", "WARN")
        return None, ""

    client = get_client(resolve_api_key(api_key))
    model = model or DEFAULT_MODEL
    voice = resolve_voice(voice, custom_voice)
    speaker1_voice = resolve_voice(speaker1_voice, None)
    speaker2_voice = resolve_voice(speaker2_voice, None)
    style = style_for_speed(style, float(speed or 1.0))

    pause_parts = split_text_on_pauses(text)
    pause_count = sum(1 for _, pause in pause_parts if pause > 0)
    if pause_count:
        log_progress(f"Found {pause_count} [pause] marker(s).", "INFO")

    final_audio_list = []
    all_notes = []

    for part_index, (segment, pause_after) in enumerate(pause_parts):
        segment = segment.strip()
        if segment:
            jobs = build_jobs(
                segment,
                model,
                voice,
                style,
                multi_speaker,
                speaker1_name,
                speaker1_voice,
                speaker2_name,
                speaker2_voice,
                clean_lowercase,
                clean_whitespace,
                clean_references,
                clean_initials,
            )
            log_progress(
                f"Part {part_index + 1}/{len(pause_parts)}: {len(jobs)} chunk(s).",
                "DEBUG",
            )
            for i, (_payload, preview) in enumerate(jobs):
                log_progress(f"  chunk {i + 1}/{len(jobs)}: {preview}", "DEBUG")
            part_audio, part_notes = generate_chunk_audio(jobs, client, parallel_chunks)
            joined = _concat_audio(part_audio, CHUNK_GAP_SEC)
            if joined is not None:
                final_audio_list.append(joined)
                all_notes.extend(part_notes)

        if pause_after > 0:
            log_progress(f"Inserting {pause_after:.2f}s pause.", "INFO")
            final_audio_list.append(silence_array(pause_after))
            all_notes.append(f"[pause {pause_after:.2f}s]")

    if not final_audio_list:
        log_progress("No audio was generated for any chunk.", "ERROR")
        return None, ""

    log_progress("Concatenating audio chunks...", "INFO")
    final_audio = np.concatenate(final_audio_list)
    notes = "\n---\n".join(all_notes)
    log_progress("Generation finished successfully.", "INFO")
    return (SAMPLE_RATE, final_audio), notes


def preview_chunks(
    text,
    voice,
    custom_voice,
    speed,
    style,
    multi_speaker,
    speaker1_name,
    speaker1_voice,
    speaker2_name,
    speaker2_voice,
    clean_lowercase,
    clean_whitespace,
    clean_references,
    clean_initials,
):
    text = (text or "").strip()
    if not text:
        return ""
    voice = resolve_voice(voice, custom_voice)
    style = style_for_speed(style, float(speed or 1.0))
    lines = [
        f"Voice: {voice}",
        f"Style: {style or '(none)'}",
        f"Mode: {'multi-speaker' if multi_speaker else 'single-speaker'}",
    ]
    pause_parts = split_text_on_pauses(text)
    chunk_i = 0
    for part_index, (segment, pause_after) in enumerate(pause_parts, start=1):
        segment = segment.strip()
        if segment:
            jobs = build_jobs(
                segment,
                DEFAULT_MODEL,
                voice,
                style,
                multi_speaker,
                speaker1_name,
                speaker1_voice,
                speaker2_name,
                speaker2_voice,
                clean_lowercase,
                clean_whitespace,
                clean_references,
                clean_initials,
            )
            for _payload, preview in jobs:
                chunk_i += 1
                lines.append(f"[{chunk_i}] {preview}")
        if pause_after > 0:
            lines.append(f"[pause {pause_after:.2f}s]")
        if part_index and not segment and pause_after:
            continue
    lines.append(f"Total chunks: {chunk_i}")
    return "\n".join(lines)


def load_text_file(filename, default_text):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError:
            return default_text
    return default_text


def get_random_quote():
    if os.path.exists("en.txt"):
        with open("en.txt", "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
            if lines:
                return random.choice(lines)
    return random.choice(RANDOM_QUOTES)


def get_zh_sample():
    return load_text_file(
        "examples/sample_zh.txt",
        "你好，世界。今天天气真好，我们一起出去走走吧。",
    )


def get_en_sample():
    return load_text_file(
        "examples/sample_en.txt",
        "Hello, this is Gemini text to speech. Choose Flash for studio narration, or Flash-Lite for fast batch conversion.",
    )


def get_pause_sample():
    return (
        "The trace is 2.521 seconds long [pause:1.5] it contains eight cores and 163 tasks.\n"
        "Wait... did you hear that? <laugh> Anyway, as I was saying."
    )


def get_dialogue_sample():
    return (
        "Joe: How's it going today Jane?\n"
        "Jane: Not too bad, how about you? Ready to test these new voices?\n"
        "Joe: Let's try a longer line, then a pause.\n"
        "Jane: Hold that thought [pause:0.8] okay, I'm ready."
    )


def load_dropped_text_file(file_path):
    if not file_path:
        return gr.update()
    path = file_path if isinstance(file_path, str) else str(file_path)
    path = path.strip().strip('"').strip("'")
    if not path or not os.path.isfile(path):
        return gr.update()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError as exc:
        log_progress(f"Failed to load dropped file {path}: {exc}", "ERROR")
        return gr.update()
    log_progress(f"Loaded text from {path} ({len(content)} chars)", "INFO")
    return content


def _to_wsl_path(path: str) -> str:
    if len(path) >= 3 and path[1] == ":" and path[0].isalpha() and path[2] in "\\/":
        drive = path[0].lower()
        rest = path[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return path.replace("\\", "/")


def _expand_path_entry(entry: str):
    import glob as _glob

    entry = _to_wsl_path(entry.strip().strip('"').strip("'"))
    if not entry or entry.startswith("#"):
        return [], None

    if any(ch in entry for ch in "*?["):
        matches = sorted(
            os.path.abspath(p)
            for p in _glob.glob(os.path.expanduser(entry))
            if os.path.isfile(p) and Path(p).suffix.lower() in TEXT_FILE_SUFFIXES
        )
        if not matches:
            return [], f"no text files matched: {entry}"
        return matches, None

    abs_entry = os.path.abspath(os.path.expanduser(entry))
    if os.path.isdir(abs_entry):
        files = []
        for child in sorted(Path(abs_entry).iterdir()):
            if child.is_file() and child.suffix.lower() in TEXT_FILE_SUFFIXES:
                files.append(str(child.resolve()))
        if not files:
            for child in sorted(Path(abs_entry).rglob("*")):
                if child.is_file() and child.suffix.lower() in TEXT_FILE_SUFFIXES:
                    files.append(str(child.resolve()))
        if not files:
            return [], f"no text files in folder: {abs_entry}"
        return files, None

    if os.path.isfile(abs_entry):
        suffix = Path(abs_entry).suffix.lower()
        name = Path(abs_entry).name.lower()
        if (
            suffix in {".list", ".files"}
            or "filelist" in name
            or name.endswith("_batch.txt")
            or name == "batch.txt"
        ):
            expanded = []
            try:
                with open(abs_entry, "r", encoding="utf-8") as handle:
                    for sub in handle:
                        sub = sub.strip().strip('"').strip("'")
                        if not sub or sub.startswith("#"):
                            continue
                        sub_files, err = _expand_path_entry(sub)
                        if sub_files:
                            expanded.extend(sub_files)
            except OSError as exc:
                return [], f"cannot read manifest {abs_entry}: {exc}"
            return expanded, None if expanded else f"empty manifest: {abs_entry}"
        if suffix and suffix not in TEXT_FILE_SUFFIXES:
            return [], f"unsupported type: {abs_entry}"
        return [abs_entry], None

    return [], f"not found: {entry}"


def _normalize_path_list(paths):
    if not paths:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    cleaned = []
    for path in paths:
        if not path:
            continue
        path = _to_wsl_path(str(path).strip().strip('"').strip("'"))
        if path:
            cleaned.append(os.path.abspath(os.path.expanduser(path)))
    return cleaned


def format_file_list(file_list):
    if not file_list:
        return "(no files added)"
    return "\n".join(f"{i + 1}. {path}" for i, path in enumerate(file_list))


def add_files_to_list(current_list, uploaded_files):
    current_list = list(current_list or [])
    added = 0
    skipped = 0
    notes = []
    for path in _normalize_path_list(uploaded_files):
        suffix = Path(path).suffix.lower()
        if suffix and suffix not in TEXT_FILE_SUFFIXES:
            skipped += 1
            notes.append(f"unsupported type: {path}")
            continue
        if not os.path.isfile(path):
            skipped += 1
            notes.append(f"not found: {path}")
            continue
        if path in current_list:
            skipped += 1
            notes.append(f"duplicate: {path}")
            continue
        current_list.append(path)
        added += 1
    status = f"Added {added} file(s). Total: {len(current_list)}."
    if skipped:
        status += f" Skipped {skipped}."
    if notes:
        status += "\n" + "\n".join(f" - {n}" for n in notes[:20])
        if len(notes) > 20:
            status += f"\n ... and {len(notes) - 20} more"
    return current_list, format_file_list(current_list), status


def add_paths_to_list(current_list, paths_text):
    current_list = list(current_list or [])
    if paths_text is None:
        paths_text = ""
    paths_text = str(paths_text)
    if not paths_text.strip():
        return (
            current_list,
            format_file_list(current_list),
            "No paths provided. Paste a file path, folder, or glob (one per line).",
        )

    collected = []
    errors = []
    for line in paths_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        files, err = _expand_path_entry(line)
        if err:
            errors.append(err)
        if files:
            collected.extend(files)

    if not collected and errors:
        return (
            current_list,
            format_file_list(current_list),
            "Nothing added:\n" + "\n".join(f" - {e}" for e in errors),
        )

    new_list, view, status = add_files_to_list(current_list, collected)
    if errors:
        status += "\nPath notes:\n" + "\n".join(f" - {e}" for e in errors)
    return new_list, view, status


def clear_file_list():
    return [], format_file_list([]), "File list cleared."


def remove_selected_files(current_list, selected_indices_text):
    current_list = list(current_list or [])
    if not current_list:
        return current_list, format_file_list(current_list), "List is empty."
    if not selected_indices_text or not selected_indices_text.strip():
        return (
            current_list,
            format_file_list(current_list),
            "Enter indices to remove (e.g. 1,3 or 2-4).",
        )

    to_remove = set()
    for token in selected_indices_text.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            try:
                start_s, end_s = token.split("-", 1)
                start, end = int(start_s), int(end_s)
                to_remove.update(range(start, end + 1))
            except ValueError:
                return current_list, format_file_list(current_list), f"Invalid range: {token}"
        else:
            try:
                to_remove.add(int(token))
            except ValueError:
                return current_list, format_file_list(current_list), f"Invalid index: {token}"

    new_list = [path for i, path in enumerate(current_list, start=1) if i not in to_remove]
    removed = len(current_list) - len(new_list)
    return new_list, format_file_list(new_list), f"Removed {removed} file(s)."


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def _to_mono(audio_data):
    audio_data = np.asarray(audio_data)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=-1)
    return np.squeeze(audio_data)


def save_audio_file(path_no_ext, sample_rate, audio_data, audio_format="wav"):
    audio_format = (audio_format or "wav").lower().lstrip(".")
    if audio_format not in {"mp3", "aac", "wav"}:
        raise ValueError(f"Unsupported audio format: {audio_format}")

    audio_data = _to_mono(audio_data)
    out_path = f"{path_no_ext}.{audio_format}"

    if audio_format == "wav" and not _ffmpeg_available():
        sf.write(out_path, audio_data, SAMPLE_RATE if sample_rate == SAMPLE_RATE else sample_rate)
        return out_path

    if audio_format in COMPRESSED_FORMATS and not _ffmpeg_available():
        raise RuntimeError(
            f"{audio_format.upper()} export requires ffmpeg on PATH. "
            "Install ffmpeg or choose WAV."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        sf.write(tmp_wav, audio_data, int(sample_rate))
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            tmp_wav,
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
        ]
        if audio_format == "mp3":
            cmd.extend(["-codec:a", "libmp3lame", "-b:a", EXPORT_BITRATE])
        elif audio_format == "aac":
            cmd.extend(["-codec:a", "aac", "-b:a", EXPORT_BITRATE, "-f", "adts"])
        else:
            cmd.extend(["-codec:a", "pcm_s16le"])
        cmd.append(out_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown ffmpeg error").strip()
            raise RuntimeError(f"ffmpeg {audio_format.upper()} encode failed: {detail}")
    finally:
        try:
            os.unlink(tmp_wav)
        except OSError:
            pass
    return out_path


def generate_with_export(
    text,
    api_key,
    model,
    voice,
    custom_voice,
    speed,
    style,
    multi_speaker,
    speaker1_name,
    speaker1_voice,
    speaker2_name,
    speaker2_voice,
    clean_lowercase,
    clean_whitespace,
    clean_references,
    clean_initials,
    parallel_chunks,
    audio_format,
):
    try:
        audio, notes = generate_first(
            text,
            api_key,
            model,
            voice,
            custom_voice,
            speed,
            style,
            multi_speaker,
            speaker1_name,
            speaker1_voice,
            speaker2_name,
            speaker2_voice,
            clean_lowercase,
            clean_whitespace,
            clean_references,
            clean_initials,
            parallel_chunks,
        )
    except Exception as exc:
        log_progress(f"Generation failed: {exc}", "ERROR")
        return None, str(exc), None
    if audio is None:
        return None, notes or "No audio generated.", None

    sample_rate, audio_data = audio
    export_dir = tempfile.mkdtemp(prefix="gemini_tts_export_")
    try:
        export_path = save_audio_file(
            os.path.join(export_dir, "gemini_tts_output"),
            sample_rate,
            audio_data,
            audio_format,
        )
    except Exception as exc:
        log_progress(f"Export failed: {exc}", "ERROR")
        return audio, notes, None
    return audio, notes, export_path


def batch_convert(
    file_list,
    api_key,
    model,
    voice,
    custom_voice,
    speed,
    style,
    multi_speaker,
    speaker1_name,
    speaker1_voice,
    speaker2_name,
    speaker2_voice,
    clean_lowercase,
    clean_whitespace,
    clean_references,
    clean_initials,
    parallel_chunks,
    output_dir,
    audio_format,
    progress=gr.Progress(track_tqdm=False),
):
    empty_player = gr.update(choices=[], value=None)
    file_list = list(file_list or [])
    if not file_list:
        return "No files in the list. Add files first.", None, empty_player, None, []

    try:
        get_client(resolve_api_key(api_key))
    except Exception as exc:
        return str(exc), None, empty_player, None, []

    audio_format = (audio_format or "wav").lower().lstrip(".")
    if audio_format in COMPRESSED_FORMATS and not _ffmpeg_available():
        return (
            f"{audio_format.upper()} export requires ffmpeg on PATH. "
            "Install ffmpeg or choose WAV.",
            None,
            empty_player,
            None,
            [],
        )

    output_dir = (output_dir or "").strip() or "batch_output"
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_progress(
        f"Batch conversion started: {len(file_list)} file(s) → {output_dir} ({audio_format})",
        "INFO",
    )
    written = []
    errors = []

    for i, path in enumerate(file_list):
        name = os.path.basename(path)
        progress((i) / len(file_list), desc=f"Converting {name}")
        log_progress(f"Batch [{i + 1}/{len(file_list)}]: {path}", "INFO")

        if not os.path.isfile(path):
            errors.append(f"{name}: file not found")
            continue

        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except Exception as exc:
            errors.append(f"{name}: read failed ({exc})")
            continue

        if not text.strip():
            errors.append(f"{name}: empty file")
            continue

        try:
            audio, _notes = generate_first(
                text,
                api_key,
                model,
                voice,
                custom_voice,
                speed,
                style,
                multi_speaker,
                speaker1_name,
                speaker1_voice,
                speaker2_name,
                speaker2_voice,
                clean_lowercase,
                clean_whitespace,
                clean_references,
                clean_initials,
                parallel_chunks,
            )
        except Exception as exc:
            errors.append(f"{name}: generation failed ({exc})")
            log_progress(f"Batch failed for {name}: {exc}", "ERROR")
            continue

        if audio is None:
            errors.append(f"{name}: no audio generated")
            continue

        sample_rate, audio_data = audio
        stem = Path(path).stem
        path_no_ext = os.path.join(output_dir, stem)
        if os.path.exists(f"{path_no_ext}.{audio_format}"):
            path_no_ext = os.path.join(output_dir, f"{stem}_{i + 1}")
        try:
            out_path = save_audio_file(path_no_ext, sample_rate, audio_data, audio_format)
            written.append(out_path)
            log_progress(f"Wrote {out_path}", "INFO")
        except Exception as exc:
            errors.append(f"{name}: write failed ({exc})")

    progress(1.0, desc="Done")
    status_lines = [
        f"Batch complete: {len(written)}/{len(file_list)} succeeded.",
        f"Format: {audio_format.upper()}",
        f"Output folder: {output_dir}",
    ]
    if written:
        status_lines.append("Files:")
        status_lines.extend(f" ✓ {p}" for p in written)
    if errors:
        status_lines.append("Errors:")
        status_lines.extend(f" ✗ {e}" for e in errors)

    status = "\n".join(status_lines)
    log_progress(status, "INFO" if not errors else "WARN")

    if not written:
        return status, None, empty_player, None, []

    choices = [(os.path.basename(p), p) for p in written]
    first = written[0]
    player_update = gr.update(choices=choices, value=first)
    return status, written, player_update, first, written


def play_batch_audio(selected_path):
    if not selected_path:
        return None
    path = str(selected_path)
    if not os.path.isfile(path):
        log_progress(f"Batch play: file not found: {path}", "WARN")
        return None
    return path


def step_batch_audio(selected_path, written_files, direction):
    written_files = list(written_files or [])
    if not written_files:
        return gr.update(), None
    try:
        idx = written_files.index(selected_path) if selected_path in written_files else 0
    except ValueError:
        idx = 0
    idx = (idx + int(direction)) % len(written_files)
    path = written_files[idx]
    return gr.update(value=path), path


def switch_model(model_id):
    spec = MODEL_PRESETS.get(model_id, MODEL_PRESETS[DEFAULT_MODEL])
    header = f"# Gemini TTS\n{spec['subtitle']}"
    return header, gr.update(info=spec["info"])


TOKEN_NOTE = """\
💡 Gemini 3.8 treats the input as a **verbatim transcript**. Put delivery in **Style**, not in the text.
🎭 Style examples: `cheerful and friendly`, `whispered urgently`, `speaking slowly`
⏸️ Timed pause: `[pause]` (0.5s) or `[pause:1.5]` (seconds, max 5)
🔉 Native tags stay in the text: `<laugh>` `<sigh>` `<pause>` `<long pause>`
💬 Ellipses (`...`) are natural hesitation — they are **not** stripped.
🗣️ Multi-speaker: one `Name: line` per line, matching Speaker 1 / Speaker 2.
🔑 Custom voices: paste a `voice_...` or `voicekey_...` ID from Voice design / replication.
"""

GEMINI_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("DM Sans"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    button_primary_background_fill="linear-gradient(135deg, #2563eb 0%, #0891b2 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #1d4ed8 0%, #0e7490 100%)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",
    shadow_drop="0 8px 24px rgba(15, 23, 42, 0.08)",
    shadow_drop_lg="0 16px 40px rgba(15, 23, 42, 0.12)",
)

APP_CSS = """
:root {
  --gemini-radius: 16px;
  --gemini-gap: 14px;
}
.gradio-container {
  max-width: 1280px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding: 18px 18px 32px !important;
  font-feature-settings: "ss01", "kern";
}
.app-header {
  align-items: center !important;
  margin-bottom: 8px;
}
.brand-title h1, .brand-title h2 {
  margin: 0 !important;
  letter-spacing: -0.03em;
  font-weight: 700 !important;
}
.brand-title p {
  margin: 4px 0 0 !important;
  opacity: 0.72;
  font-size: 0.95rem;
}
.theme-toggle {
  justify-content: flex-end !important;
  align-items: center !important;
}
.theme-toggle button {
  min-width: 88px;
  border-radius: 999px !important;
}
.settings-panel, .main-panel, .batch-card {
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
  border-radius: var(--gemini-radius) !important;
  padding: 14px 14px 8px !important;
  box-shadow: var(--shadow-drop);
}
.gradio-container .tabs {
  margin-top: 4px;
}
button.primary, .generate-row button {
  border-radius: 12px !important;
}
#theme-light, #theme-dark {
  border-radius: 999px !important;
}
.dark #theme-dark,
#theme-light {
  font-weight: 700 !important;
}
.dark #theme-light {
  font-weight: 500 !important;
}
.dark .settings-panel,
.dark .main-panel,
.dark .batch-card {
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
}
footer { display: none !important; }
"""

APP_JS = """
function initGeminiTheme() {
  const apply = (mode) => {
    try { localStorage.setItem('gemini-tts-theme', mode); } catch (e) {}
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== mode) {
      url.searchParams.set('__theme', mode);
      window.location.replace(url.href);
    }
  };
  try {
    const url = new URL(window.location);
    const current = url.searchParams.get('__theme');
    if (!current) {
      const saved = localStorage.getItem('gemini-tts-theme');
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      apply(saved || (prefersDark ? 'dark' : 'light'));
    } else {
      localStorage.setItem('gemini-tts-theme', current);
    }
  } catch (e) {}
  return 'theme-ready';
}
"""

THEME_LIGHT_JS = """
() => {
  try { localStorage.setItem('gemini-tts-theme', 'light'); } catch (e) {}
  const url = new URL(window.location);
  url.searchParams.set('__theme', 'light');
  window.location.href = url.href;
}
"""

THEME_DARK_JS = """
() => {
  try { localStorage.setItem('gemini-tts-theme', 'dark'); } catch (e) {}
  const url = new URL(window.location);
  url.searchParams.set('__theme', 'dark');
  window.location.href = url.href;
}
"""

_startup = MODEL_PRESETS[DEFAULT_MODEL]
_has_env_key = bool(resolve_api_key())

with gr.Blocks(title="Gemini TTS") as app:
    with gr.Row(elem_classes="app-header"):
        with gr.Column(scale=5, elem_classes="brand-title"):
            header_md = gr.Markdown(
                f"# Gemini TTS\n{_startup['subtitle']}"
            )
        with gr.Column(scale=2, min_width=220, elem_classes="theme-toggle"):
            with gr.Row():
                light_btn = gr.Button("Light", elem_id="theme-light", variant="secondary")
                dark_btn = gr.Button("Dark", elem_id="theme-dark", variant="secondary")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=280, elem_classes="settings-panel"):
            gr.Markdown("### Voice")
            api_key = gr.Textbox(
                label="Gemini API key",
                type="password",
                placeholder="GEMINI_API_KEY" if _has_env_key else "Paste key or set GEMINI_API_KEY",
                info="Uses GEMINI_API_KEY from the environment when this field is empty.",
            )
            model_select = gr.Dropdown(
                MODEL_DROPDOWN,
                value=DEFAULT_MODEL,
                label="Model",
                info=_startup["info"],
            )
            voice = gr.Dropdown(
                PREBUILT_VOICES,
                value=DEFAULT_VOICE,
                label="Voice",
                filterable=True,
                info="30 prebuilt studio voices. Language is detected automatically.",
            )
            custom_voice = gr.Textbox(
                label="Custom voice ID (optional)",
                placeholder="voice_... or voicekey_...",
                info="Overrides the dropdown when set.",
            )
            speed = gr.Slider(
                minimum=0.5,
                maximum=2,
                value=1,
                step=0.1,
                label="Speed",
                info="Mapped to style (speaking slowly / rapidly). 1.0 leaves pacing unchanged.",
            )
            style = gr.Textbox(
                label="Style",
                lines=2,
                placeholder="cheerful and friendly",
                info="Turn-level delivery. Leave empty unless you need a specific performance.",
            )
            audio_format = gr.Dropdown(
                choices=["wav", "mp3", "aac"],
                value="wav",
                label="Output Format",
                info="24 kHz / mono. WAV is 16-bit PCM. MP3 and AAC need ffmpeg.",
            )
            with gr.Accordion("Multi-speaker", open=False):
                multi_speaker = gr.Checkbox(
                    label="Enable two-speaker dialogue",
                    value=False,
                    info="Transcript must use Name: line format matching the names below.",
                )
                speaker1_name = gr.Textbox(label="Speaker 1 name", value="Joe")
                speaker1_voice = gr.Dropdown(
                    PREBUILT_VOICES, value="Puck", label="Speaker 1 voice", filterable=True
                )
                speaker2_name = gr.Textbox(label="Speaker 2 name", value="Jane")
                speaker2_voice = gr.Dropdown(
                    PREBUILT_VOICES, value="Kore", label="Speaker 2 voice", filterable=True
                )
            with gr.Accordion("Text cleaning", open=False):
                clean_lowercase = gr.Checkbox(label="Convert to lowercase", value=False)
                clean_whitespace = gr.Checkbox(label="Normalize whitespace", value=True)
                clean_references = gr.Checkbox(
                    label="Remove reference numbers",
                    value=True,
                    info="Strips footnote digits like end.3. Decimal numbers such as 2.521 are kept.",
                )
                clean_initials = gr.Checkbox(label="Format initials (J.R.R.)", value=True)
            with gr.Accordion("Performance", open=False):
                parallel_chunks = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Parallel chunks",
                    info="Concurrent Gemini API calls. Lower this if you hit rate limits.",
                )

        with gr.Column(scale=3, elem_classes="main-panel"):
            with gr.Tabs():
                with gr.Tab("Single"):
                    single_file = gr.File(
                        label="Drop a .txt or .md file",
                        file_count="single",
                        file_types=[".txt", ".md", ".text", ".markdown"],
                        type="filepath",
                    )
                    text = gr.Textbox(
                        label="Input text",
                        lines=8,
                        value=(
                            "Hello, this is Gemini text to speech. "
                            "Choose Flash for studio narration, or Flash-Lite for fast batch conversion."
                        ),
                    )
                    with gr.Row():
                        zh_sample_btn = gr.Button("中文示例", variant="secondary")
                        random_btn = gr.Button("Random quote", variant="secondary")
                        en_sample_btn = gr.Button("English sample", variant="secondary")
                        pause_sample_btn = gr.Button("Pause demo", variant="secondary")
                        dialogue_btn = gr.Button("Dialogue", variant="secondary")
                    with gr.Row(elem_classes="generate-row"):
                        generate_btn = gr.Button("Generate", variant="primary")
                        stop_generate_btn = gr.Button("Stop", variant="stop")
                    out_audio = gr.Audio(label="Output", interactive=False)
                    out_file = gr.File(label="Download", interactive=False)
                    with gr.Accordion("Chunks / request preview", open=False):
                        out_ps = gr.Textbox(interactive=False, show_label=False, lines=6)
                        tokenize_btn = gr.Button("Preview chunks", variant="secondary")
                    gr.Markdown(TOKEN_NOTE)

                with gr.Tab("Batch"):
                    gr.Markdown(
                        "Add `.txt` / `.md` files by upload, folder path, or glob "
                        "(`~/text/*.txt`). Uses the voice settings on the left."
                    )
                    batch_files_state = gr.State([])
                    batch_results_state = gr.State([])
                    with gr.Row(equal_height=True):
                        with gr.Column(elem_classes="batch-card"):
                            batch_upload = gr.File(
                                label="Add text files",
                                file_count="multiple",
                                file_types=[".txt", ".md", ".text", ".markdown"],
                                type="filepath",
                            )
                            batch_paths = gr.Textbox(
                                label="File or folder paths (one per line)",
                                lines=4,
                                placeholder="/home/kuoping/text\n/path/to/chapter1.txt\n/path/to/*.md",
                            )
                            with gr.Row():
                                add_paths_btn = gr.Button("Add paths", variant="secondary")
                                clear_list_btn = gr.Button("Clear", variant="secondary")
                            batch_file_list_view = gr.Textbox(
                                label="Queue",
                                lines=8,
                                value=format_file_list([]),
                                interactive=False,
                            )
                            with gr.Row():
                                remove_indices = gr.Textbox(
                                    label="Remove by index",
                                    placeholder="1,3 or 2-4",
                                    scale=3,
                                )
                                remove_btn = gr.Button("Remove", variant="secondary", scale=1)

                        with gr.Column(elem_classes="batch-card"):
                            batch_output_dir = gr.Textbox(
                                label="Output folder",
                                value=str(Path("batch_output").resolve()),
                            )
                            with gr.Row():
                                batch_convert_btn = gr.Button("Batch convert", variant="primary")
                                stop_batch_btn = gr.Button("Stop", variant="stop")
                            batch_status = gr.Textbox(label="Status", lines=8, interactive=False)
                            batch_outputs = gr.File(
                                label="Generated files",
                                file_count="multiple",
                                interactive=False,
                            )
                            with gr.Row():
                                batch_play_select = gr.Dropdown(
                                    label="Preview",
                                    choices=[],
                                    value=None,
                                    interactive=True,
                                )
                                batch_prev_btn = gr.Button("Prev", variant="secondary", scale=0)
                                batch_next_btn = gr.Button("Next", variant="secondary", scale=0)
                            batch_audio = gr.Audio(
                                label="Player",
                                interactive=False,
                                autoplay=True,
                            )

    model_select.change(fn=switch_model, inputs=[model_select], outputs=[header_md, model_select])
    light_btn.click(fn=None, js=THEME_LIGHT_JS)
    dark_btn.click(fn=None, js=THEME_DARK_JS)
    single_file.change(fn=load_dropped_text_file, inputs=[single_file], outputs=[text])
    zh_sample_btn.click(fn=get_zh_sample, inputs=[], outputs=[text])
    random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
    en_sample_btn.click(fn=get_en_sample, inputs=[], outputs=[text])
    pause_sample_btn.click(fn=get_pause_sample, inputs=[], outputs=[text])
    dialogue_btn.click(fn=get_dialogue_sample, inputs=[], outputs=[text])

    generation_inputs = [
        text,
        api_key,
        model_select,
        voice,
        custom_voice,
        speed,
        style,
        multi_speaker,
        speaker1_name,
        speaker1_voice,
        speaker2_name,
        speaker2_voice,
        clean_lowercase,
        clean_whitespace,
        clean_references,
        clean_initials,
        parallel_chunks,
        audio_format,
    ]
    preview_inputs = [
        text,
        voice,
        custom_voice,
        speed,
        style,
        multi_speaker,
        speaker1_name,
        speaker1_voice,
        speaker2_name,
        speaker2_voice,
        clean_lowercase,
        clean_whitespace,
        clean_references,
        clean_initials,
    ]

    generation_event = generate_btn.click(
        fn=generate_with_export,
        inputs=generation_inputs,
        outputs=[out_audio, out_ps, out_file],
    )
    tokenize_btn.click(fn=preview_chunks, inputs=preview_inputs, outputs=[out_ps])
    stop_generate_btn.click(fn=None, cancels=generation_event)

    batch_upload.change(
        fn=add_files_to_list,
        inputs=[batch_files_state, batch_upload],
        outputs=[batch_files_state, batch_file_list_view, batch_status],
    )
    add_paths_btn.click(
        fn=add_paths_to_list,
        inputs=[batch_files_state, batch_paths],
        outputs=[batch_files_state, batch_file_list_view, batch_status],
    )
    clear_list_btn.click(
        fn=clear_file_list,
        inputs=[],
        outputs=[batch_files_state, batch_file_list_view, batch_status],
    )
    remove_btn.click(
        fn=remove_selected_files,
        inputs=[batch_files_state, remove_indices],
        outputs=[batch_files_state, batch_file_list_view, batch_status],
    )

    batch_inputs = [
        batch_files_state,
        api_key,
        model_select,
        voice,
        custom_voice,
        speed,
        style,
        multi_speaker,
        speaker1_name,
        speaker1_voice,
        speaker2_name,
        speaker2_voice,
        clean_lowercase,
        clean_whitespace,
        clean_references,
        clean_initials,
        parallel_chunks,
        batch_output_dir,
        audio_format,
    ]
    batch_event = batch_convert_btn.click(
        fn=batch_convert,
        inputs=batch_inputs,
        outputs=[batch_status, batch_outputs, batch_play_select, batch_audio, batch_results_state],
    )
    stop_batch_btn.click(fn=None, cancels=batch_event)
    batch_play_select.change(
        fn=play_batch_audio,
        inputs=[batch_play_select],
        outputs=[batch_audio],
    )
    batch_prev_btn.click(
        fn=lambda path, files: step_batch_audio(path, files, -1),
        inputs=[batch_play_select, batch_results_state],
        outputs=[batch_play_select, batch_audio],
    )
    batch_next_btn.click(
        fn=lambda path, files: step_batch_audio(path, files, 1),
        inputs=[batch_play_select, batch_results_state],
        outputs=[batch_play_select, batch_audio],
    )


def _find_free_port(preferred=7860, fallback_start=17860, tries=100):
    candidates = [preferred, *range(fallback_start, fallback_start + tries)]
    for port in candidates:
        with socket.socket() as sock:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(
        f"Cannot find an empty port among {preferred} or "
        f"{fallback_start}-{fallback_start + tries - 1}."
    )


if __name__ == "__main__":
    import signal

    env_port = os.getenv("GRADIO_SERVER_PORT")
    server_port = None
    if env_port:
        candidate = int(env_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", candidate))
                server_port = candidate
        except OSError:
            print(
                f"WARN: Port {candidate} from GRADIO_SERVER_PORT is unavailable; "
                "selecting a free port instead."
            )
    if server_port is None:
        server_port = _find_free_port()
    print(f"DEBUG: Launching Gradio on port {server_port}")
    allowed_paths = [
        str(Path.cwd().resolve()),
        str(Path.home().resolve()),
        tempfile.gettempdir(),
    ]

    _interrupt_count = {"n": 0}

    def _handle_interrupt(signum, frame):
        _interrupt_count["n"] += 1
        if _interrupt_count["n"] == 1:
            print("\nPress Ctrl-C again to close the app.", flush=True)
            return
        print("\nClosing.", flush=True)
        os._exit(0)

    signal.signal(signal.SIGINT, _handle_interrupt)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_interrupt)

    print("DEBUG: Press Ctrl-C twice to close the app.")
    app.queue().launch(
        inbrowser=True,
        server_port=server_port,
        allowed_paths=allowed_paths,
        theme=GEMINI_THEME,
        css=APP_CSS,
        js=APP_JS,
    )
