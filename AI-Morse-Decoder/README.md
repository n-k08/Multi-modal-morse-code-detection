# Morse Code Audio Decoder

A real-time morse code decoder that listens to a microphone or reads an audio file, detects the tone using FFT analysis, adaptively determines the speed, and decodes the characters. Optionally connects to Claude AI to stream a corrected version of the decoded text alongside the raw output.

```
====================================================================================================
  Morse Decoder  |  Ctrl+C to stop
====================================================================================================
RAW: FOR GOD SO LOVED THE WORLD HE GAVE HIS ONLY BEGOTTEN SON , THAT WHOSOEVER
 AI: FOR GOD SO LOVED THE WORLD, HE GAVE HIS ONLY BEGOTTEN SON, THAT WHOSOEVER
```

---

## Installation

```bash
pip install -r requirements.txt
```

For MP3, AAC, or M4A file support, also install pydub and ffmpeg:

```bash
pip install pydub
# macOS:  brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

---

## Usage

### Decode a live microphone input

```bash
python main.py
```

### List available audio input devices

```bash
python main.py --list-devices
```

### Decode from a specific microphone

```bash
python main.py --device "Focusrite"    # partial name match, case-insensitive
python main.py --device 2              # device index from --list-devices
```

### Decode an audio file

```bash
python main.py --file recording.wav
python main.py --file contest.mp3      # requires pydub + ffmpeg
```

---

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--file FILE` | — | Audio file to decode (WAV, FLAC, OGG; MP3/AAC with pydub+ffmpeg) |
| `--device DEVICE` | system default | Input device name (partial match) or index |
| `--list-devices` | — | Print available audio input devices and exit |
| `--freq HZ` | auto-detect | Morse tone frequency in Hz |
| `--wpm N` | 15 | Initial WPM estimate; adapts automatically |
| `--no-ai` | — | Disable AI correction (raw decode only) |
| `--ai-model MODEL` | claude-opus-4-6 | Claude model to use for correction |
| `--sample-rate HZ` | 44100 | Audio sample rate |
| `--ai-debounce SEC` | 0.8 | Seconds of silence after a word before triggering AI correction |
| `--threshold 0–1` | 0.35 | Tone detection threshold. Lower (e.g. 0.12) for weak or noisy signals |
| `--hold-off MS` | 40.0 | Milliseconds of silence required before declaring tone OFF. Raise to filter transition glitches; lower for very fast CW |

---

## **Test Files**

**morse1.wav** is a W1AW morse code practice file from ARRL at 10 WPM. The correct text is in morse1.txt. Note: The raw decode prints each character separately because W1AW sends 10 WPM code practice using the Farnsworth method — characters are sent at ~18–20 WPM with gaps between letters and words stretched out to slow the overall speed to 10 WPM. The AI-corrected output groups them into proper words.

**morse2.wav** and **morse3.wav** are audio files from **Pixabay.com** (Special Effects)

**EXAMPLE:**

```
$ python main.py --file morse3.wav

RAW: ERYONE [..--]HO7ELIEVESSNBIMOILLNOT[---.]ERIS[....-...]UTE&A[...-..--..-.--.-.][.-.....--...-......]
 AI: THIS WAY HE GAVE HIS ONE AND ONLY SON, SO THAT EVERYONE WHO BELIEVES IN HIM WILL NOT PERISH BUT HAVE


Decoded text:
  FOR GOD SO LOVED THE WORLD IN THIS WAY HE GAVE HIS ONE AND ONLY SON, SO THAT EVERYONE WHO BELIEVES IN HIM WILL NOT PERISH BUT HAVE

Raw decode:
  FOR GOD HO LOVEDTHE OORLDIN THIS[..--]AY HE GAVE HISJNE ANDONLY SON, DOTHAT EVERYONE [..--]HO7ELIEVESSNBIMOILLNOT[---.]ERIS[....-...]UTE&A[...-..--..-.--.-.][.-.....--...-......]

Detected frequency: 800 Hz
Final speed estimate: 9.9 WPM
```



## Connecting to Claude AI

The AI correction layer uses the [Anthropic API](https://www.anthropic.com). To enable it:

**1. Get an API key**

Sign up at [console.anthropic.com](https://console.anthropic.com) and create an API key.

**2. Add the key to a `.env` file**

Create a file named `.env` in the same directory as `main.py` and add your key:

```
ANTHROPIC_API_KEY="sk-ant-..."
```

The program reads this file automatically on startup. You can also set the key as an environment variable instead:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**3. Run the decoder**

```bash
python main.py --file recording.wav
```

The AI line will appear below the raw decode line, streaming token by token roughly one word behind the raw output.

**Choosing a model**

The default model is `claude-opus-4-6` (most capable). For faster or cheaper operation:

```bash
python main.py --file recording.wav --ai-model claude-haiku-4-5
```

**Disabling AI**

```bash
python main.py --file recording.wav --no-ai
```

If `ANTHROPIC_API_KEY` is not set in `.env` or the environment, the program automatically falls back to `--no-ai` with a warning.

---

## How It Works

The program is an asyncio pipeline with five stages:

```
AudioCapture / FileAudioSource
    → ToneDetector      (FFT-based, per 10ms chunk)
        → TimingDecoder (adaptive state machine)
            → Display   (raw line, instant)
            → AICorrector (Claude streaming, debounced)
                → Display (AI line, ~1 word behind)
```

### 1. Audio Input

**Live capture** (`AudioCapture`) opens a sounddevice input stream. The sounddevice callback runs in a separate thread and pushes each audio chunk into an asyncio queue, which the main event loop reads from. This keeps the audio thread non-blocking.

**File input** (`FileAudioSource`) loads the entire file in a thread pool executor so the event loop is not blocked. Supported natively: WAV, FLAC, OGG, AIFF. MP3/AAC/M4A fall back to pydub, which requires ffmpeg in `PATH`. If the file's sample rate differs from `--sample-rate`, it is resampled using `scipy.signal.resample_poly` (polyphase filter, accurate and efficient).

Timestamps for file input are derived from sample position (`frame / sample_rate`) rather than wall-clock time. This means the adaptive timing logic works correctly regardless of how fast the file is processed.

### 2. Tone Detection (`ToneDetector`)

Each 10 ms audio chunk is processed with a Hanning-windowed FFT. The detector looks for a strong single tone in the 300–1500 Hz range.

**Auto-detect mode**: On the first strong detection, the dominant frequency in the 300–1500 Hz band is locked in as the tracked frequency, then gently adapted via exponential moving average (rate 0.05) on subsequent detections.

**Pinned mode** (`--freq`): Monitors a fixed ±40 Hz band around the specified frequency.

**Two-condition gate**: A chunk is declared ON only when both conditions hold:

1. *Ratio gate* — band power / total spectral power exceeds the threshold (default 0.35). This ensures the target frequency is the dominant spectral feature rather than noise.
2. *Absolute power gate* — band power exceeds 25% of the observed peak band power. This prevents false positives in near-silence, where even tiny residual energy at the target frequency can produce a high ratio.

**Hold-off hysteresis** (`--hold-off`, default 40 ms): The tone is not declared OFF until the signal has been continuously absent for at least this long. This prevents brief amplitude dips — common in recorded audio and weak signals — from being misread as inter-element gaps. The OFF event timestamp is set to when the silence *began*, not when the hold-off expired, so element durations are measured accurately.

### 3. Timing Decoder (`TimingDecoder`)

Converts a stream of `('on', t)` / `('off', t)` events into characters using the ITU morse timing model:

| Element | Duration |
|---------|----------|
| Dot | 1 unit |
| Dash | 3 units |
| Intra-character gap | 1 unit |
| Inter-letter gap | 3 units |
| Word gap | 7 units |

**Initial unit**: `1200 ms / WPM` (PARIS standard). At the default 15 WPM, one unit is 80 ms.

**Adaptive timing**: After each element, the unit estimate is updated from a rolling window of the last 12 observed dot or dash durations. This allows the decoder to track transmitters that vary in speed.

**Artifact filtering**: Very short elements are discarded before they can corrupt the unit estimate. The minimum plausible element is `max(15 ms, 0.3 × unit)`. Additionally, dots shorter than `0.5 × unit` are counted as dots but excluded from the adaptive estimate — they are treated as transition artifacts from keying transients.

**Timeout**: `check_timeout()` is called on every audio chunk. If the gap since the last OFF event exceeds the word-gap threshold, any pending character is flushed and a word space is emitted. This handles end-of-transmission silences.

**Lookup**: Decoded dot/dash patterns are looked up in a standard ITU morse table covering letters, digits, and punctuation. Unrecognized patterns are output as `[pattern]` for inspection.

### 4. AI Correction (`AICorrector`)

After each word gap, a debounced coroutine is scheduled (default 0.8 s delay). If another word arrives within the debounce window, the timer resets. When the timer fires, the full accumulated raw text is sent to the Claude API using the streaming messages endpoint.

The system prompt primes Claude with amateur radio domain knowledge: Q-codes (QSL, QTH, QRM, …), prosigns (AR, SK, BK, …), and common abbreviations (CQ, DE, 73, ES, …). Claude outputs only the corrected text with no commentary.

Tokens stream back one by one and are appended to the AI display line in real time. If a new correction is triggered before the previous one finishes streaming, the in-flight API request is cancelled and a fresh one begins.

At end-of-file, any pending debounce is cancelled and a final correction is run over the complete decoded text, awaited with a 30-second timeout.

### 5. Display

The terminal display uses ANSI escape codes to redraw two lines in place without scrolling. `\033[2A` moves the cursor up two lines; `\033[2K\r` clears each line before rewriting it. Lines are truncated to 100 characters so they never wrap. The RAW line updates on every decoded character; the AI line updates on every streamed token from Claude.

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, CLI argument parsing, asyncio pipeline wiring |
| `audio.py` | `AudioCapture` (live mic) and `FileAudioSource` (file input) |
| `tone_detector.py` | FFT-based tone detection with hold-off hysteresis |
| `timing_decoder.py` | Adaptive morse timing state machine |
| `ai_corrector.py` | Claude API streaming integration |
| `display.py` | Two-line ANSI terminal display |
| `morse_table.py` | ITU morse code lookup table |
| `requirements.txt` | Python dependencies |
