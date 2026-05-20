"""
Scanner (Lexer): consumes characters from SourceReader one at a time and produces Tokens on demand.
"""

from .source_reader import SourceReader
from .token import Token, TokenType
from .exceptions import LexicalError, ConstraintError, ConstraintError


SINGLE_CHAR_TOKENS = {
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "^": TokenType.CARET,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    ",": TokenType.COMMA,
    "=": TokenType.EQUALS,
}


class Scanner:
    def __init__(self, reader: SourceReader, max_var_len: int = None):
        self._reader = reader
        self._max_var_len = max_var_len
        self._lookahead = None      # one-character buffer (already read)
        self._lookahead_ready = False
        # Prime the pump
        self._advance()

    # Internal character helpers

    def _advance(self):
        """Read next char from reader into lookahead buffer."""
        self._lookahead = self._reader.next_char()
        self._lookahead_ready = True

    def _peek(self):
        """Return the current lookahead character without consuming it."""
        return self._lookahead

    def _consume(self):
        """Return and consume the current lookahead, then read next."""
        ch = self._lookahead
        self._advance()
        return ch

    # Public API

    def next_token(self) -> Token:
        """Return the next Token from the input stream."""
        # Skip whitespace (but not newlines)
        while self._peek() is not None and self._peek() in (" ", "\t", "\r"):
            self._consume()

        ch = self._peek()

        # EOF
        if ch is None:
            return Token(TokenType.EOF, None,
                         self._reader.line, self._reader.col)

        line = self._reader.line
        col = self._reader.col

        # Newline -> statement separator
        if ch == "\n":
            self._consume()
            return Token(TokenType.NEWLINE, "\n", line, col)

        # Comment: # until end of line
        if ch == "#":
            while self._peek() is not None and self._peek() != "\n":
                self._consume()
            return self.next_token()

        # Single-character tokens
        if ch in SINGLE_CHAR_TOKENS:
            self._consume()
            return Token(SINGLE_CHAR_TOKENS[ch], ch, line, col)

        # Number literal: digits and optional decimal point
        if ch.isdigit() or ch == ".":
            return self._scan_number(line, col)

        # Identifier or keyword
        if ch.isalpha() or ch == "_":
            return self._scan_identifier(line, col)

        # Unknown character
        self._consume()
        raise LexicalError(
            f"Unexpected character '{ch}'", line, col
        )

    # Sub-scanners

    def _scan_number(self, line, col) -> Token:
        """
        Scan a numeric literal: integer or floating point.
        Reads one character at a time; no regex or readlines.
        """
        digits = []
        has_dot = False

        while True:
            ch = self._peek()
            if ch is None:
                break
            if ch.isdigit():
                digits.append(self._consume())
            elif ch == "." and not has_dot:
                # Make sure it's followed by a digit
                has_dot = True
                digits.append(self._consume())
            else:
                break

        raw = "".join(digits)
        if raw == "." or raw == "":
            raise LexicalError(
                f"Malformed numeric literal '{raw}'", line, col
            )
        # Second decimal point coverege (e.g. '1.2.3'). Lexical error.
        if self._peek() == ".":
            raise LexicalError(
                f"Malformed numeric literal: unexpected '.' after '{raw}'",
                self._reader.line, self._reader.col,
            )
        return Token(TokenType.NUMBER, float(raw), line, col)

    def _scan_identifier(self, line, col) -> Token:
        """
        Scan an identifier: letter/underscore followed by letters/digits/underscores.
        Enforces max_var_len if set (checked later in semantic phase, but length
        is captured here for error reporting).
        """
        chars = []
        while True:
            ch = self._peek()
            if ch is None:
                break
            if ch.isalnum() or ch == "_":
                chars.append(self._consume())
            else:
                break

        name = "".join(chars)
        if self._max_var_len is not None and len(name) > self._max_var_len:
            raise ConstraintError(
                f"Variable name '{name}' exceeds maximum length of {self._max_var_len} characters",
                line, col,
            )
        return Token(TokenType.IDENTIFIER, name, line, col)