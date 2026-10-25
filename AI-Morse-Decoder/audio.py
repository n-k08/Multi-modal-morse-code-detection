import asyncio
import os
import time
from math import gcd
from typing import AsyncIterator, Optional, Tuple

import numpy as np
import sounddevice as sd


def list_devices() -> None:
    """Print all available audio input devices."""
    print("\nAvailable audio input devices:")
    print("-" * 60)
    default_in = sd.default.device[0]
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            marker = " ← default" if i == default_in else ""
            print(f"  [{i:2d}] {dev['name']}{marker}")
            print(f"       Channels: {dev['max_input_channels']}, "
                  f"Rate: {int(dev['default_samplerate'])} Hz")
    print()


class AudioCapture:
    """
    Captures mono audio from a named (or indexed) input device.
    Yields (chunk, timestamp) pairs as an async generator.
    """

    def __init__(
        self,
        device_name: Optional[str] = None,
        sample_rate: int = 44100,
        chunk_ms: int = 10,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_ms / 1000)
        self._device_idx = self._resolve_device(device_name)
        self._queue: asyncio.Queue = None
        self._loop: asyncio.AbstractEventLoop = None
        self._stream: sd.InputStream = None

    def _resolve_device(self, name: Optional[str]) -> Optional[int]:
        if name is None:
            return None
        # Try numeric index first
        try:
            idx = int(name)
            dev = sd.query_devices(idx)
            if dev['max_input_channels'] > 0:
                print(f"Using device [{idx}]: {dev['name']}")
                return idx
        except (ValueError, sd.PortAudioError):
            pass
        # Search by name substring (case-insensitive)
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0 and name.lower() in dev['name'].lower():
                print(f"Using device [{i}]: {dev['name']}")
                return i
        raise ValueError(
            f"No input device matching '{name}'. "
            f"Run --list-devices to see available options."
        )

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        chunk = indata[:, 0].copy()
        if self._loop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)
            except asyncio.QueueFull:
                pass  # Drop if consumer is too slow

    async def __aenter__(self) -> "AudioCapture":
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=300)
        self._stream = sd.InputStream(
            device=self._device_idx,
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            dtype=np.float32,
            callback=self._callback,
        )
        self._stream.start()
        return self

    async def __aexit__(self, *_) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()

    async def chunks(self) -> AsyncIterator[Tuple[np.ndarray, float]]:
        """Yield (audio_chunk, monotonic_timestamp) continuously."""
        while True:
            chunk = await self._queue.get()
            yield chunk, time.monotonic()


# ---------------------------------------------------------------------------
# File audio source
# ---------------------------------------------------------------------------

def _read_audio_file(path: str) -> Tuple[np.ndarray, int]:
    """
    Read any audio file and return (mono_float32_array, sample_rate).

    Tries soundfile, then PyAV for compressed formats. Pydub is a final
    fallback for formats unsupported by the installed PyAV codecs.
    """
    # --- soundfile (no external binary needed) ---
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        mono = data.mean(axis=1)
        return mono.astype(np.float32), sr
    except Exception:
        pass

    # --- PyAV (bundled FFmpeg codecs; no system FFmpeg executable needed) ---
    av_error = None
    try:
        import av

        with av.open(path) as container:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise ValueError("The file contains no audio stream.")
            sample_rate = stream.codec_context.sample_rate or stream.rate
            if not sample_rate:
                raise ValueError("The audio stream has no sample rate.")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
            decoded_frames = []
            for frame in container.decode(stream):
                decoded_frames.extend(resampler.resample(frame))
            decoded_frames.extend(resampler.resample(None))
            if not decoded_frames:
                raise ValueError("The audio stream is empty.")
            mono = np.concatenate([frame.to_ndarray().reshape(-1) for frame in decoded_frames])
            return mono.astype(np.float32), int(sample_rate)
    except Exception as exc:
        av_error = exc

    # --- pydub (optional final fallback) ---
    try:
        from pydub import AudioSegment  # type: ignore
    except ImportError:
        ext = os.path.splitext(path)[1].lower()
        raise RuntimeError(
            f"Cannot read '{path}' (extension: {ext}).\n"
            f"Audio decoding failed: {av_error}"
        ) from av_error

    try:
        seg = AudioSegment.from_file(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read '{path}': {exc}") from exc

    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if seg.channels > 1:
        samples = samples.reshape(-1, seg.channels).mean(axis=1)
    # pydub gives int16-range values; normalise to [-1, 1]
    samples /= 32768.0
    return samples.astype(np.float32), sr


def _resample(data: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample a mono float32 array from from_sr to to_sr using polyphase filter."""
    from scipy.signal import resample_poly  # type: ignore
    g = gcd(from_sr, to_sr)
    return resample_poly(data, to_sr // g, from_sr // g).astype(np.float32)


class FileAudioSource:
    """
    Reads an audio file (WAV, FLAC, OGG, MP3, …) and yields chunks exactly
    like AudioCapture does, using audio-time timestamps (not wall clock).

    Because timestamps are derived from sample position / sample_rate, the
    timing decoder's adaptive logic works correctly at any processing speed.
    The file is consumed as fast as the event loop allows — no artificial sleep.

    Supported formats (no extra deps):   WAV, FLAC, OGG/Vorbis, AIFF, …
    Requires pydub + ffmpeg in PATH for: MP3, AAC, M4A, WMA, …
    """

    def __init__(
        self,
        file_path: str,
        target_sample_rate: int = 44100,
        chunk_ms: int = 10,
    ):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: '{file_path}'")
        self.file_path = file_path
        self.target_sample_rate = target_sample_rate
        self.chunk_size = int(target_sample_rate * chunk_ms / 1000)
        self._audio: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "FileAudioSource":
        # Load in a thread so we don't block the event loop
        loop = asyncio.get_running_loop()
        self._audio = await loop.run_in_executor(None, self._load)
        return self

    async def __aexit__(self, *_) -> None:
        self._audio = None

    def _load(self) -> np.ndarray:
        data, sr = _read_audio_file(self.file_path)
        if sr != self.target_sample_rate:
            data = _resample(data, sr, self.target_sample_rate)
        return data

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def duration_seconds(self) -> Optional[float]:
        if self._audio is None:
            return None
        return len(self._audio) / self.target_sample_rate

    @property
    def total_chunks(self) -> Optional[int]:
        if self._audio is None:
            return None
        return (len(self._audio) + self.chunk_size - 1) // self.chunk_size

    # ------------------------------------------------------------------
    # Chunk generator
    # ------------------------------------------------------------------

    async def chunks(self) -> AsyncIterator[Tuple[np.ndarray, float]]:
        """
        Yield (chunk, audio_timestamp_seconds).
        Yields to the event loop every chunk so async tasks (AI, display) can run.
        """
        audio = self._audio
        n = len(audio)
        pos = 0
        while pos < n:
            end = min(pos + self.chunk_size, n)
            chunk = audio[pos:end]
            # Pad last chunk if shorter than chunk_size
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
            timestamp = pos / self.target_sample_rate
            yield chunk, timestamp
            pos = end
            # Yield control so the event loop can handle AI / display tasks
            await asyncio.sleep(0)
