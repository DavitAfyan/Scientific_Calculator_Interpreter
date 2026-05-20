"""
SourceReader: reads exactly one character at a time from a file or stdin.
Tracks line and column position for error reporting.
"""


class SourceReader:
    def __init__(self, stream):
        self._stream = stream
        self._line = 1
        self._col = 0
        self._current = None
        self._exhausted = False

    @property
    def line(self):
        return self._line

    @property
    def col(self):
        return self._col

    def next_char(self):
        """Read and return exactly one character, or None at EOF."""
        if self._exhausted:
            return None
        ch = self._stream.read(1)
        if ch == "":
            self._exhausted = True
            return None
        if ch == "\n":
            self._line += 1
            self._col = 0
        else:
            self._col += 1
        self._current = ch
        return ch

    def is_exhausted(self):
        return self._exhausted
