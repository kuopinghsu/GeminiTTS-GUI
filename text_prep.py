"""Text cleaning, pause splitting, and chunking for Gemini TTS."""

from __future__ import annotations

import re

PAUSE_MARKER_RE = re.compile(
    r"\[pause(?::\s*(\d+(?:\.\d+)?)\s*s?)?\]",
    re.IGNORECASE,
)
DIALOGUE_RE = re.compile(
    r"^([A-Za-z][\w .'\-]{0,40}|[\u4e00-\u9fff]{1,12})[：:]\s*(.*)$"
)
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

# Stay well under Gemini 3.8 TTS input (8192) and output (16384) token caps.
MAX_CHARS_LATIN = 2800
MAX_CHARS_CJK = 1400


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text.strip())


def remove_inline_reference_numbers(text: str) -> str:
    """Remove footnote-style numbers after punctuation (e.g. end.3).

    Keep decimal values such as 2.521 or .5.
    """

    def _keep_or_strip(match):
        punct = match.group(1)
        if punct == ".":
            i = match.start()
            prev = text[i - 1] if i > 0 else ""
            if prev.isdigit() or prev.isspace() or prev == "":
                return match.group(0)
        return punct

    return re.sub(r'([.!?,\"\'”’)\]])(\d+)(?=\s|$)', _keep_or_strip, text)


def replace_letter_period_sequences(text: str) -> str:
    """Convert 'J.R.R.' to 'J R R'."""

    def replacer(match):
        cleaned = match.group(0).rstrip(".")
        letters = cleaned.split(".")
        return " ".join(letters)

    return re.sub(r"\b(?:[A-Za-z]\.){2,}", replacer, text)


def _pause_seconds(match: re.Match) -> float:
    seconds = float(match.group(1)) if match.group(1) else 0.5
    return min(5.0, max(0.15, seconds))


def split_text_on_pauses(text: str) -> list[tuple[str, float]]:
    """Split on ``[pause]`` / ``[pause:1.5]``.

    Returns ``[(segment, pause_after_seconds), ...]``. Ellipses stay in the
    transcript so Gemini can treat them as natural hesitation; timed pauses
    become real silence after synthesis.
    """
    pieces = []
    last = 0
    for match in PAUSE_MARKER_RE.finditer(text):
        pieces.append((text[last:match.start()], _pause_seconds(match)))
        last = match.end()
    pieces.append((text[last:], 0.0))
    return pieces


def looks_cjk(text: str) -> bool:
    if not text:
        return False
    return len(CJK_RE.findall(text)) >= max(1, len(text) // 8)


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if looks_cjk(text):
        parts = re.split(r"(?<=[。！？!?．…\n])", text)
        sentences = [p.strip() for p in parts if p.strip()]
        return sentences or [text]
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences or [text]


def split_long_sentence(sentence: str, max_len: int, seps: list[str] | None = None) -> list[str]:
    if seps is None:
        seps = [";", ":", "—", "-", ",", "，", "；", " "]

    sentence = sentence.strip()
    if len(sentence) <= max_len:
        return [sentence]

    if not seps:
        return [sentence[i:i + max_len].strip() for i in range(0, len(sentence), max_len)]

    sep = seps[0]
    parts = sentence.split(sep)
    if len(parts) == 1:
        return split_long_sentence(sentence, max_len, seps=seps[1:])

    chunks = []
    current = parts[0].strip()
    for part in parts[1:]:
        candidate = (current + sep + part).strip()
        if len(candidate) > max_len:
            chunks.extend(split_long_sentence(current, max_len, seps=seps[1:]))
            current = part.strip()
        else:
            current = candidate
    if current:
        chunks.extend(split_long_sentence(current, max_len, seps=seps[1:]))
    return [c for c in chunks if c]


def group_sentences(sentences: list[str], max_chars: int, joiner: str = " ") -> list[str]:
    chunks = []
    current_chunk = []
    current_length = 0
    sep_len = len(joiner)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_len = len(sentence)
        extra = sep_len if current_chunk else 0
        if sentence_len > max_chars:
            if current_chunk:
                chunks.append(joiner.join(current_chunk))
            chunks.extend(split_long_sentence(sentence, max_chars))
            current_chunk = []
            current_length = 0
        elif current_length + sentence_len + extra <= max_chars:
            current_chunk.append(sentence)
            current_length += sentence_len + extra
        else:
            chunks.append(joiner.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len

    if current_chunk:
        chunks.append(joiner.join(current_chunk))
    return chunks


def prepare_text(
    text: str,
    clean_lowercase: bool,
    clean_whitespace: bool,
    clean_references: bool,
    clean_initials: bool,
) -> list[str]:
    if clean_lowercase and not looks_cjk(text):
        text = text.lower()
    if clean_whitespace:
        text = normalize_whitespace(text)
    if clean_references:
        text = remove_inline_reference_numbers(text)
    if clean_initials and not looks_cjk(text):
        text = replace_letter_period_sequences(text)

    sentences = split_sentences(text)
    cjk = looks_cjk(text)
    max_chars = MAX_CHARS_CJK if cjk else MAX_CHARS_LATIN
    joiner = "" if cjk else " "
    return group_sentences(sentences, max_chars=max_chars, joiner=joiner)


def parse_dialogue_turns(text: str) -> list[tuple[str, str]] | None:
    """Parse ``Name: line`` turns. Returns None if the text is not dialogue."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    turns = []
    for line in lines:
        match = DIALOGUE_RE.match(line)
        if not match:
            return None
        speaker = match.group(1).strip()
        utterance = match.group(2).strip()
        if not speaker or not utterance:
            return None
        turns.append((speaker, utterance))
    speakers = {name for name, _ in turns}
    if len(speakers) < 2:
        return None
    return turns


def estimate_tokens(text: str) -> int:
    cjk = len(CJK_RE.findall(text))
    other = max(0, len(text) - cjk)
    return int(cjk * 1.2 + other / 3.5) + 8


def style_for_speed(base_style: str, speed: float) -> str:
    style = (base_style or "").strip()
    extras = []
    if speed <= 0.7:
        extras.append("speaking very slowly")
    elif speed < 0.95:
        extras.append("speaking slowly")
    elif speed >= 1.4:
        extras.append("speaking very rapidly")
    elif speed > 1.05:
        extras.append("speaking rapidly")
    if not extras:
        return style
    extra = ", ".join(extras)
    return f"{style}, {extra}" if style else extra
