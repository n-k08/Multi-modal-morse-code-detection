from typing import Optional, Tuple

import numpy as np


class ToneDetector:
    """
    Detects morse code keying from audio chunks using FFT.

    If target_freq is given, monitors that specific frequency band.
    Otherwise, auto-detects: tracks whatever strong single tone appears
    in the 300–1500 Hz range first.

    Returns ('on', timestamp) / ('off', timestamp) on state transitions.

    hold_off_ms: the signal must be continuously absent for at least this
    many milliseconds before an OFF event is emitted.  This prevents
    brief amplitude dips in the tone (common in audio files and weak
    signals) from being misread as inter-element gaps.
    """

    FREQ_MIN = 300    # Hz — lower bound for morse tones
    FREQ_MAX = 1500   # Hz — upper bound for morse tones
    BAND_WIDTH = 80   # Hz — ±40 Hz around tracked frequency
    ADAPT_RATE = 0.05 # EMA rate for frequency tracking (slow — keeps stable)

    # Signal must exceed this fraction of total spectral power to count as ON.
    # 0.35 works well for clean recordings and strong signals.
    # Lower to ~0.12 for weak or noisy signals.
    SIGNAL_THRESHOLD = 0.35

    def __init__(
        self,
        sample_rate: int = 44100,
        target_freq: Optional[float] = None,
        hold_off_ms: float = 20.0,
        threshold: Optional[float] = None,
    ):
        if threshold is not None:
            self.SIGNAL_THRESHOLD = threshold
        self.sample_rate = sample_rate
        self.target_freq = target_freq        # user-pinned freq (never adapted)
        self._tracked_freq: Optional[float] = target_freq  # working estimate
        self._is_on = False

        # Hysteresis: require this many consecutive off-chunks before
        # emitting an OFF event.  Calibrated on first chunk.
        self._hold_off_ms = hold_off_ms
        self._hold_off_chunks: int = 0   # set on first process() call
        self._off_streak: int = 0        # current run of off-chunks
        self._off_started: Optional[float] = None  # timestamp when off streak began
        self._peak_band_power: float = 0.0

    @property
    def detected_freq(self) -> Optional[float]:
        return self._tracked_freq

    def process(self, chunk: np.ndarray, timestamp: float) -> Optional[Tuple[str, float]]:
        """
        Process one audio chunk.
        Returns ('on', t) or ('off', t) on a state change, else None.
        """
        # Calibrate hold counter on first call
        if self._hold_off_chunks == 0:
            chunk_ms = len(chunk) / self.sample_rate * 1000
            self._hold_off_chunks = max(1, round(self._hold_off_ms / chunk_ms))

        raw_on = self._detect(chunk)

        if raw_on:
            self._off_streak = 0
            if not self._is_on:
                self._is_on = True
                self._off_started = None
                return ('on', timestamp)
        else:
            if self._off_streak == 0:
                self._off_started = timestamp   # record when gap began
            self._off_streak += 1
            if self._is_on and self._off_streak >= self._hold_off_chunks:
                self._is_on = False
                return ('off', self._off_started)

        return None

    def _detect(self, chunk: np.ndarray) -> bool:
        n = len(chunk)
        windowed = chunk * np.hanning(n)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
        total_power = np.sum(spectrum) + 1e-10

        if self._tracked_freq is not None:
            half = self.BAND_WIDTH / 2
            mask = (freqs >= self._tracked_freq - half) & (freqs <= self._tracked_freq + half)
            if not np.any(mask):
                return False
            band_power = np.sum(spectrum[mask])

            # Update peak band power (slow decay so it tracks signal level)
            if band_power > self._peak_band_power:
                self._peak_band_power = band_power
            else:
                self._peak_band_power *= 0.9999  # very slow decay

            # Two-condition gate:
            # 1. Band fraction of total spectrum must exceed threshold
            #    (ensures it IS the dominant frequency, not noise)
            # 2. Absolute band power must exceed 25% of observed peak
            #    (rejects false positives in near-silence where ratio stays high)
            ratio_ok = band_power / total_power > self.SIGNAL_THRESHOLD
            power_ok = (self._peak_band_power == 0 or
                        band_power > self._peak_band_power * 0.25)

            if ratio_ok and power_ok:
                if self.target_freq is None:
                    peak_freq = freqs[mask][np.argmax(spectrum[mask])]
                    self._tracked_freq = (
                        (1 - self.ADAPT_RATE) * self._tracked_freq
                        + self.ADAPT_RATE * peak_freq
                    )
                return True
            return False

        # Auto-detect mode
        mask = (freqs >= self.FREQ_MIN) & (freqs <= self.FREQ_MAX)
        if not np.any(mask):
            return False
        peak_idx = np.argmax(spectrum[mask])
        peak_power = spectrum[mask][peak_idx]
        if peak_power / total_power > self.SIGNAL_THRESHOLD:
            self._tracked_freq = freqs[mask][peak_idx]
            self._peak_band_power = peak_power
            return True
        return False
