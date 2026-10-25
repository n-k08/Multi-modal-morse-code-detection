from typing import Callable, List, Optional

from morse_table import MORSE_CODE


class TimingDecoder:
    """
    Converts morse keying events (on/off + timestamp) into text.

    Timing model (all relative to one "unit"):
        dot  = 1 unit   dash  = 3 units
        intra-char gap = 1 unit
        inter-letter gap = 3 units
        word gap = 7 units

    The unit duration adapts in real time from observed dot/dash lengths.
    Starting estimate comes from the --wpm argument.

    Callbacks (all optional, called synchronously):
        on_char(char)   — fired on each decoded character or space
        on_word_space() — fired additionally when a word gap is detected
    """

    ADAPT_WINDOW = 12  # rolling window size for dot/dash averaging

    def __init__(self, wpm: int = 15, gap_scale: float = 1.0, strict_protocol: bool = False):
        # 1 unit = 1200 ms / WPM  (PARIS standard)
        self._unit: float = 1.2 / wpm
        self._gap_scale = max(1.0, gap_scale)
        self._strict_protocol = strict_protocol

        # Rolling history for adaptive timing
        self._dot_durs: List[float] = []
        self._dash_durs: List[float] = []

        # State
        self._key_start: Optional[float] = None
        self._gap_start: Optional[float] = None
        self._was_on = False
        self._current_pattern = ""   # dots/dashes accumulating for this letter

        # Output
        self._raw_chars: List[str] = []
        self.on_char: Optional[Callable[[str], None]] = None
        self.on_word_space: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def process_event(self, event_type: str, timestamp: float) -> None:
        """Feed a keying event produced by ToneDetector."""
        if event_type == 'on':
            self._was_on = True
            if self._gap_start is not None:
                gap_dur = timestamp - self._gap_start
                self._handle_gap(gap_dur)
                self._gap_start = None
            self._key_start = timestamp

        elif event_type == 'off':
            self._was_on = False
            if self._key_start is not None:
                key_dur = timestamp - self._key_start
                self._handle_element(key_dur)
                self._key_start = None
            self._gap_start = timestamp

    def flush(self) -> None:
        """Force-emit any character/space still pending (call at end of input)."""
        if self._current_pattern:
            self._emit_letter()

    def check_timeout(self, now: float) -> None:
        """
        Call on every audio chunk to flush any letter/word that has been
        sitting in a gap longer than the word threshold.
        """
        if self._gap_start is not None and not self._was_on:
            if now - self._gap_start > self._word_gap_threshold:
                if self._current_pattern:
                    self._emit_letter()
                    self._emit_space()
                self._gap_start = None

    @property
    def raw_text(self) -> str:
        return ''.join(self._raw_chars)

    @property
    def current_wpm(self) -> float:
        return round(1.2 / self._unit, 1) if self._unit > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Threshold helpers                                                     #
    # ------------------------------------------------------------------ #

    @property
    def _dot_threshold(self) -> float:
        return 2.0 * self._unit

    @property
    def _letter_gap_threshold(self) -> float:
        return 2.0 * self._unit * self._gap_scale

    @property
    def _word_gap_threshold(self) -> float:
        return 5.0 * self._unit * self._gap_scale

    # ------------------------------------------------------------------ #
    # Internal logic                                                        #
    # ------------------------------------------------------------------ #

    def _handle_element(self, duration: float) -> None:
        # Reject very short elements — these are transition glitches, not
        # real dots/dashes.  The minimum plausible dot is ~0.3× the current
        # unit estimate (so at 20 WPM / 60ms unit, we reject anything <18ms).
        min_dur = max(0.015, 0.3 * self._unit)
        if duration < min_dur:
            return

        if duration < self._dot_threshold:
            self._current_pattern += '.'
            # Only feed reasonable dot durations into the adaptive estimate.
            # Dots that are suspiciously short (< half current unit) are
            # transition artifacts that would pull the unit too low.
            if duration >= 0.5 * self._unit:
                self._dot_durs.append(duration)
                if len(self._dot_durs) > self.ADAPT_WINDOW:
                    self._dot_durs.pop(0)
        else:
            self._current_pattern += '-'
            self._dash_durs.append(duration)
            if len(self._dash_durs) > self.ADAPT_WINDOW:
                self._dash_durs.pop(0)
        self._adapt_unit()

    def _handle_gap(self, duration: float) -> None:
        if duration >= self._word_gap_threshold:
            self._emit_letter()
            self._emit_space()
        elif duration >= self._letter_gap_threshold:
            self._emit_letter()
        # else: intra-char gap — keep building the pattern

    def _emit_letter(self) -> None:
        if not self._current_pattern:
            return
        char = MORSE_CODE.get(self._current_pattern)
        if char is None and self._strict_protocol:
            self._current_pattern = ""
            return
        if char is None:
            char = f'[{self._current_pattern}]'
        self._current_pattern = ""
        self._raw_chars.append(char)
        if self.on_char:
            self.on_char(char)

    def _emit_space(self) -> None:
        self._raw_chars.append(' ')
        if self.on_char:
            self.on_char(' ')
        if self.on_word_space:
            self.on_word_space()

    def _adapt_unit(self) -> None:
        """Update unit estimate from observed element durations."""
        if self._dot_durs:
            self._unit = sum(self._dot_durs) / len(self._dot_durs)
        elif self._dash_durs:
            self._unit = (sum(self._dash_durs) / len(self._dash_durs)) / 3.0
