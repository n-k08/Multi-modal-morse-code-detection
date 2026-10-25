import sys


class Display:
    """
    Three-line terminal display that updates in place using ANSI escape codes.

        Status: File: recording.wav  |  50%  30s / 60s  |  Freq: 700 Hz
        RAW:  CQ CQ DE W_TSU PSE K      ← updates instantly, char by char
         AI:  CQ CQ DE W5TSU PSE K      ← streams in ~1 word behind

    Call start() once, then update_raw() / update_ai() / info() freely.
    All three lines are always redrawn in place — no scrolling.
    """

    WIDTH = 100  # characters before truncation

    def __init__(self):
        self._raw = ""
        self._ai = ""
        self._status = ""
        self._active = False

    def start(self) -> None:
        print()
        print("=" * self.WIDTH)
        print("  Morse Decoder  |  Ctrl+C to stop")
        print("=" * self.WIDTH)
        sys.stdout.write("\n")          # status line
        sys.stdout.write("RAW: \n")     # raw line
        sys.stdout.write(" AI: \n")     # ai line
        sys.stdout.flush()
        self._active = True

    def update_raw(self, text: str) -> None:
        self._raw = self._tail(text)
        self._redraw()

    def update_ai(self, text: str) -> None:
        self._ai = self._tail(text)
        self._redraw()

    def info(self, msg: str) -> None:
        """Update the status line without disrupting the RAW/AI lines."""
        self._status = self._tail(msg)
        if not self._active:
            print(msg)
            return
        self._redraw()

    def stop(self) -> None:
        if self._active:
            sys.stdout.write('\n')
            sys.stdout.flush()
            self._active = False

    def _redraw(self) -> None:
        if not self._active:
            return
        sys.stdout.write('\033[3A')                          # move up 3 lines
        sys.stdout.write(f'\033[2K\r{self._status}\n')      # status line
        sys.stdout.write(f'\033[2K\rRAW: {self._raw}\n')    # raw line
        sys.stdout.write(f'\033[2K\r AI: {self._ai}\n')     # ai line
        sys.stdout.flush()

    def _tail(self, text: str) -> str:
        """Keep only the last WIDTH characters so lines don't wrap."""
        return text[-self.WIDTH:] if len(text) > self.WIDTH else text
