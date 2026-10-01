# Gemini TTS Local Web UI

A local [Gradio](https://www.gradio.app/) interface for [Gemini 3.8 Flash TTS](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash-tts) and [Gemini 3.8 Flash-Lite TTS](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash-lite-tts). Long text is split into chunks, synthesized through the Gemini API, and concatenated. Switch checkpoints in the UI:

* **Flash** — `gemini-3.8-flash-tts`: studio-grade narration, acting, dialects (130 languages)
* **Flash-Lite** — `gemini-3.8-flash-lite-tts`: faster, cheaper high-volume production (101 languages)

Layout and workflow follow [KokoroTTS-GUI-Extended](https://github.com/kuopinghsu/KokoroTTS-GUI-Extended) (Single + Batch tabs, pause markers, WAV/MP3 export). Synthesis runs in the cloud via the Gemini API rather than a local model.

See the [Gemini text-to-speech guide](https://ai.google.dev/gemini-api/docs/speech-generation) for prompting, voices, and `speech_metadata`.

## Features

* **Two models** — switch in the UI:
  * **Flash** (`gemini-3.8-flash-tts`): maximum fidelity, acting nuance, dialect coverage
  * **Flash-Lite** (`gemini-3.8-flash-lite-tts`): high throughput and lower cost
* **30 prebuilt studio voices** — Zephyr, Puck, Kore, Fenrir, …
* **Custom voice IDs** — paste a `voice_...` (Voice design / replication) or `voicekey_...`
* **Style field** — turn-level delivery (`cheerful and friendly`, `whispered urgently`). Gemini 3.8 treats the transcript as verbatim, so stage directions belong here, not in the text.
* **Single Text tab** — type or drag-and-drop a `.txt` / `.md` file
* **Batch Files tab** — build a file list (upload, paste paths, or a whole folder) and convert many files at once
* **WAV / MP3 / AAC export** — MP3 and AAC need `ffmpeg`
* **Batch preview** — play completed files with Prev/Next after a batch run
* **Parallel chunks** — long text is split and requested concurrently (watch API rate limits)
* **Two-speaker dialogue** — `Name: line` transcripts with two configured voices
* **Text preprocessing** — lowercase, whitespace normalize, reference-number removal, initials formatting
* **Pauses** — `[pause]` or `[pause:1.5]` insert real silence. Native tags such as `<laugh>` / `<pause>` stay in the transcript.
* **Chunk preview** — inspect how text will be split before you spend API credits
* **Sample library** — 中文示例, Random Quote, English sample, Pause demo, Dialogue

## Prerequisites

1. **Python 3.10+**
2. **Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey)
3. **ffmpeg** (optional, for MP3 / AAC) — must include `libmp3lame`
   * **Linux:** `sudo apt-get install ffmpeg`
   * **Mac:** `brew install ffmpeg`
   * **Windows:** install from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to `PATH`

## Installation

1. **Clone or open this project**, then create a virtual environment:

   ```bash
   uv venv --python 3.12 .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   uv pip install -r requirements.txt
   ```

3. **Set your API key** (any of these works):

   ```bash
   export GEMINI_API_KEY="your-key"
   ```

   Or copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`. You can also paste the key in the UI.

## Usage

1. Activate the environment and start the app:

   ```bash
   source .venv/bin/activate
   uv run python app.py
   ```

2. The UI opens in your browser. Shared controls (model, voice, style, cleaning, parallel chunks, output format) apply to both tabs.

3. **Choose a model** with the **Model** dropdown (left panel):
   * **Flash** — audiobooks, complex dialogue, dialects
   * **Flash-Lite** — batch conversion and everyday read-aloud  
   Then pick a **Voice**, optional **Style**, and **Speed**.

### Single Text

* Enter text, or drag a `.txt` / `.md` file onto the drop zone
* Click **Generate** — preview plays in the UI; a downloadable WAV/MP3 appears below
* Use **Preview chunks** to inspect how long text will be split

### Style, tags, and pauses

Gemini 3.8 TTS reads the input as a **verbatim transcript**. Put sustained delivery in **Style**. Keep momentary vocal events as angle-bracket tags in the text.

| Control | Effect |
|---|---|
| Style field | Turn-level emotion / pace / whisper, e.g. `calm narrator` |
| Speed slider | Adds `speaking slowly` / `speaking rapidly` to Style when not 1.0 |
| `[pause]` | 0.5s of real silence (inserted after synthesis) |
| `[pause:1.5]` | 1.5s silence (`[pause:2s]` also works; min 0.15s, max 5s) |
| `...` | Natural hesitation (left in the transcript) |
| `<laugh>` `<sigh>` `<pause>` | Native Gemini vocal-burst / pause tags |

```text
The trace is 2.521 seconds long [pause:1.5] it contains eight cores and 163 tasks.
Wait... did you hear that? <laugh> Anyway, as I was saying.
```

Commas, semicolons, and a single period only change intonation; they are not a timed pause. Extra spaces or blank lines are collapsed when **Normalize whitespace** is on.

Decimal numbers such as `2.521` are kept even with **Remove reference numbers** enabled (that option only strips footnote-style digits like `paper.3`).

### Multi-speaker

1. Open **Multi-speaker** and enable two-speaker dialogue
2. Set Speaker 1 / Speaker 2 names to match the transcript
3. Use one `Name: line` per line:

```text
Joe: How's it going today Jane?
Jane: Not too bad, how about you? Ready to test these new voices?
```

The Gemini API supports two speakers per request. Each turn must name a configured speaker.

### Batch Files

1. Add files via upload, or paste paths (one per line):
   * a single file: `/path/to/chapter1.txt`
   * a **folder**: `/path/to/text` (adds all `.txt` / `.md` inside)
   * a glob: `/path/to/*.txt`
2. Click **Add paths** (or use Clear / Remove by index)
3. Set the **Output folder** and **Output Format** (`wav`, `mp3`, or `aac`)
4. Click **Batch convert**
5. When finished, use **Preview** / Prev / Next to listen to results

## Configuration

| Control | Description |
|---|---|
| **Model** | `gemini-3.8-flash-tts` or `gemini-3.8-flash-lite-tts` |
| **Voice** | One of 30 prebuilt studio voices |
| **Custom voice ID** | Optional `voice_...` / `voicekey_...` (overrides Voice) |
| **Style** | `speech_metadata.style` for the whole turn |
| **Speed** | 0.5×–2.0× mapped onto style pacing (not a DSP time-stretch) |
| **Output Format** | wav, mp3, or aac |
| **Text Cleaning** | Lowercase, whitespace, references, initials |
| **Parallel chunks** | 1–5 concurrent API requests (lower if you hit 429s) |
| **GRADIO_SERVER_PORT** | Optional env var to force a port; if that port cannot bind, the app picks a free one |

On some WSL/Windows setups, ports in the 7000–9000 range are reserved. The app prefers `7860`, then falls back (e.g. around `17860`).

Long scripts are split to stay under the Gemini TTS limits (8,192 input tokens and 16,384 output tokens per request).

## Troubleshooting

* **Missing API key:** Set `GEMINI_API_KEY`, add it to `.env`, or paste it in the UI. Create a key in [Google AI Studio](https://aistudio.google.com/apikey).
* **429 / rate limit:** Lower **Parallel chunks** to 1 and retry. The app already retries transient 429/5xx errors.
* **MP3 export fails:** Install `ffmpeg` with `libmp3lame`, or switch **Output Format** to `wav`.
* **Gradio `InvalidPathError` when returning batch files:** The app allows serving files under your home directory. Restart after updating so `allowed_paths` is applied. Outputs are still written to your chosen folder even if the download widget fails.
* **Port already in use / empty port range:** Unset a bad `GRADIO_SERVER_PORT`, or set one that can bind, e.g. `GRADIO_SERVER_PORT=17860 python app.py`.
* **Style is spoken aloud:** You are putting directions in the transcript. Use the **Style** field instead. Gemini 3.8 reads text verbatim.
* **Upgrade SDK:** This app needs `google-genai >= 2.25.0` for the Interactions API used by Gemini 3.8 TTS.

```bash
uv run python -m unittest tests.test_text_prep
```

## Project structure

```text
├── app.py                 # Gradio UI (single + batch)
├── text_prep.py           # Cleaning, pauses, chunking
├── requirements.txt       # Python dependencies
├── examples/              # Optional sample text
├── tests/                 # Unit tests (no API key required)
├── batch_output/          # Default batch output folder (created on use)
└── .venv/                 # Virtual environment (created during install)
```

## License

Apache License 2.0. This project calls the Gemini API; use of the models is subject to [Google Gemini API terms](https://ai.google.dev/gemini-api/terms).

### Acknowledgements

* Layout and batch/pause workflow adapted from [KokoroTTS-GUI-Extended](https://github.com/kuopinghsu/KokoroTTS-GUI-Extended)
* [Gemini 3.8 Flash TTS](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-8-text-to-speech/) and the [Gemini speech generation guide](https://ai.google.dev/gemini-api/docs/speech-generation)
