"""
Real-time Morse Code Audio Decoder with AI correction.

Pipeline:
    AudioCapture (device by name/index)  ─┐
    FileAudioSource (WAV/MP3/FLAC/…)    ─┘
        → ToneDetector (FFT, auto-detect freq)
            → TimingDecoder (adaptive WPM, state machine)
                → Display (raw line, instant)
                → AICorrector (Claude streaming, debounced)
                    → Display (AI line, ~1 word behind)
"""

import asyncio
import argparse
import sys
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from audio import list_devices, AudioCapture, FileAudioSource
from tone_detector import ToneDetector
from timing_decoder import TimingDecoder
from ai_corrector import AICorrector
from display import Display


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time morse code audio decoder with AI correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py --list-devices
  python main.py --device "Focusrite"
  python main.py --device 2 --freq 700 --wpm 20
  python main.py --file recording.wav
  python main.py --file contest.mp3 --freq 700 --wpm 25
  python main.py --no-ai
        """,
    )
    p.add_argument("--file", "-i", default=None,
                   help="Audio file to decode (WAV, FLAC, OGG, MP3*, …) "
                        "*MP3 requires: pip install pydub  +  ffmpeg in PATH")
    p.add_argument("--device", "-d", default=None,
                   help="Input device name (partial match) or index")
    p.add_argument("--list-devices", "-l", action="store_true",
                   help="List available audio input devices and exit")
    p.add_argument("--freq", "-f", type=float, default=None,
                   help="Tone frequency Hz (default: auto-detect)")
    p.add_argument("--wpm", "-w", type=int, default=15,
                   help="Initial WPM estimate, adapts automatically (default: 15)")
    p.add_argument("--no-ai", action="store_true",
                   help="Disable AI correction (raw decode only)")
    p.add_argument("--ai-model", default="claude-opus-4-6",
                   help="Claude model for correction (default: claude-opus-4-6)")
    p.add_argument("--sample-rate", type=int, default=44100,
                   help="Audio sample rate Hz (default: 44100)")
    p.add_argument("--ai-debounce", type=float, default=0.8,
                   help="Seconds to wait after last word before calling AI (default: 0.8)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Tone detection threshold 0–1 (default: 0.35 for clean signals, "
                        "try 0.12 for weak/noisy audio)")
    p.add_argument("--hold-off", type=float, default=40.0,
                   help="ms of silence required before declaring tone OFF (default: 40). "
                        "Raise to filter transition glitches; lower for very fast CW")
    return p


async def run(args: argparse.Namespace) -> None:
    display = Display()

    # --- AI corrector (optional) ---
    ai_corrector: Optional[AICorrector] = None
    if not args.no_ai:
        try:
            ai_corrector = AICorrector(model=args.ai_model)
        except ValueError as exc:
            print(f"Warning: {exc}", file=sys.stderr)
            print("Continuing without AI correction.\n", file=sys.stderr)

    # --- Core pipeline components ---
    tone_detector = ToneDetector(
        sample_rate=args.sample_rate,
        target_freq=args.freq,
        threshold=args.threshold,
        hold_off_ms=args.hold_off,
    )
    timing_decoder = TimingDecoder(wpm=args.wpm)

    # --- AI debounce state ---
    ai_text = ""
    _correction_handle: Optional[asyncio.Task] = None
    _debounce_handle: Optional[asyncio.Task] = None

    async def run_correction() -> None:
        """Stream AI correction for the current raw text."""
        nonlocal ai_text, _correction_handle
        if _correction_handle and not _correction_handle.done():
            _correction_handle.cancel()

        raw = timing_decoder.raw_text
        if not raw.strip() or ' ' not in raw:
            return  # Need at least one full word before correcting

        async def stream_correction() -> None:
            nonlocal ai_text
            ai_text = ""
            try:
                async for token in ai_corrector.correct(raw):
                    ai_text += token
                    display.update_ai(ai_text)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                msg = getattr(exc, 'body', {})
                if isinstance(msg, dict):
                    msg = msg.get('error', {}).get('message', str(exc))
                else:
                    msg = getattr(exc, 'message', str(exc)).strip('"')
                print(f"\nAI error: {msg}", file=sys.stderr)

        _correction_handle = asyncio.create_task(stream_correction())

    async def debounced_correction() -> None:
        """Wait debounce period, then run correction if no new words arrived."""
        await asyncio.sleep(args.ai_debounce)
        await run_correction()

    def on_char(char: str) -> None:
        display.update_raw(timing_decoder.raw_text)

    def on_word_space() -> None:
        """Fired on word gaps — schedule a debounced AI correction."""
        nonlocal _debounce_handle
        if ai_corrector is None:
            return
        if _debounce_handle and not _debounce_handle.done():
            _debounce_handle.cancel()
        _debounce_handle = asyncio.create_task(debounced_correction())

    timing_decoder.on_char = on_char
    timing_decoder.on_word_space = on_word_space

    # --- Build audio source ---
    if args.file:
        source = FileAudioSource(
            file_path=args.file,
            target_sample_rate=args.sample_rate,
        )
        source_label = f"File: {args.file}"
    else:
        source = AudioCapture(
            device_name=args.device,
            sample_rate=args.sample_rate,
        )
        source_label = "Source: live microphone"

    # --- Start display ---
    display.start()
    freq_label = f"{args.freq:.0f} Hz" if args.freq else "auto-detect"
    display.info(f"{source_label}  |  Freq: {freq_label}  |  Init: {args.wpm} WPM")

    # --- Main loop ---
    try:
        async with source:
            total = getattr(source, "total_chunks", None)
            chunk_num = 0
            async for chunk, timestamp in source.chunks():
                event = tone_detector.process(chunk, timestamp)
                if event:
                    timing_decoder.process_event(event[0], event[1])
                timing_decoder.check_timeout(timestamp)

                # Progress indicator for file mode
                if total:
                    chunk_num += 1
                    if chunk_num % 500 == 0:
                        pct = chunk_num / total * 100
                        elapsed = timestamp
                        dur = source.duration_seconds
                        display.info(
                            f"{source_label}  |  "
                            f"{pct:.0f}%  {elapsed:.0f}s / {dur:.0f}s  |  "
                            f"Freq: {tone_detector.detected_freq:.0f} Hz"
                            if tone_detector.detected_freq else
                            f"{source_label}  |  {pct:.0f}%  {elapsed:.0f}s / {dur:.0f}s"
                        )

            # Flush any character still pending at end of file
            if args.file:
                timing_decoder.flush()
                display.update_raw(timing_decoder.raw_text)
                # Run a final AI correction over the complete text
                if ai_corrector and timing_decoder.raw_text.strip():
                    if _debounce_handle and not _debounce_handle.done():
                        _debounce_handle.cancel()
                    await run_correction()
                    # Wait for the final correction to finish
                    if _correction_handle:
                        try:
                            await asyncio.wait_for(_correction_handle, timeout=30)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass

    except KeyboardInterrupt:
        pass
    finally:
        for task in [_debounce_handle, _correction_handle]:
            if task and not task.done():
                task.cancel()
        display.stop()
        raw = timing_decoder.raw_text
        if raw.strip():
            if ai_text.strip():
                print(f"\nDecoded text:\n  {ai_text}")
                print(f"\nRaw decode:\n  {raw}")
            else:
                print(f"\nDecoded text:\n  {raw}")
        if tone_detector.detected_freq:
            print(f"\nDetected frequency: {tone_detector.detected_freq:.0f} Hz")
        else:
            print("\nNo tone detected.")
        print(f"Final speed estimate: {timing_decoder.current_wpm} WPM")


def main() -> None:
    args = build_parser().parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.file and args.device:
        print("Error: --file and --device are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print("Input file not found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
