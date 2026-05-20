"""
Unit tests for the Scanner (Lexer).
"""
import io
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from calculator.src.source_reader import SourceReader
from calculator.src.scanner import Scanner
from calculator.src.token import TokenType
from calculator.src.exceptions import LexicalError


def make_scanner(text: str, max_var_len=None) -> Scanner:
    return Scanner(SourceReader(io.StringIO(text)), max_var_len=max_var_len)


def all_tokens(text: str):
    sc = make_scanner(text)
    tokens = []
    while True:
        t = sc.next_token()
        tokens.append(t)
        if t.type == TokenType.EOF:
            break
    return tokens


# ── Numeric literals ──────────────────────────────────────────────────

class TestNumberTokens:
    def test_integer(self):
        toks = all_tokens("42")
        assert toks[0].type == TokenType.NUMBER
        assert toks[0].value == 42.0

    def test_float(self):
        toks = all_tokens("3.14159")
        assert toks[0].type == TokenType.NUMBER
        assert abs(toks[0].value - 3.14159) < 1e-9

    def test_leading_dot(self):
        toks = all_tokens(".5")
        assert toks[0].type == TokenType.NUMBER
        assert toks[0].value == 0.5

    def test_zero(self):
        toks = all_tokens("0")
        assert toks[0].type == TokenType.NUMBER
        assert toks[0].value == 0.0

    def test_large_number(self):
        toks = all_tokens("1000000")
        assert toks[0].value == 1_000_000.0

    def test_number_followed_by_operator(self):
        toks = all_tokens("5+3")
        assert toks[0].type == TokenType.NUMBER
        assert toks[1].type == TokenType.PLUS
        assert toks[2].type == TokenType.NUMBER


# ── Identifiers ───────────────────────────────────────────────────────

class TestIdentifierTokens:
    def test_simple_identifier(self):
        toks = all_tokens("abc")
        assert toks[0].type == TokenType.IDENTIFIER
        assert toks[0].value == "abc"

    def test_long_identifier(self):
        toks = all_tokens("long_var_name")
        assert toks[0].type == TokenType.IDENTIFIER
        assert toks[0].value == "long_var_name"

    def test_identifier_with_digits(self):
        toks = all_tokens("var1")
        assert toks[0].type == TokenType.IDENTIFIER
        assert toks[0].value == "var1"

    def test_underscore_start(self):
        toks = all_tokens("_private")
        assert toks[0].type == TokenType.IDENTIFIER
        assert toks[0].value == "_private"

    def test_single_letter(self):
        toks = all_tokens("x")
        assert toks[0].type == TokenType.IDENTIFIER
        assert toks[0].value == "x"


# ── Operators & delimiters ────────────────────────────────────────────

class TestOperatorTokens:
    def test_all_operators(self):
        toks = all_tokens("+-*/^")
        expected = [TokenType.PLUS, TokenType.MINUS, TokenType.STAR,
                    TokenType.SLASH, TokenType.CARET]
        assert [t.type for t in toks[:5]] == expected

    def test_parens_and_comma(self):
        toks = all_tokens("(,)")
        assert toks[0].type == TokenType.LPAREN
        assert toks[1].type == TokenType.COMMA
        assert toks[2].type == TokenType.RPAREN

    def test_equals(self):
        toks = all_tokens("=")
        assert toks[0].type == TokenType.EQUALS


# ── Whitespace & newlines ─────────────────────────────────────────────

class TestWhitespace:
    def test_spaces_skipped(self):
        toks = all_tokens("  42  ")
        assert toks[0].type == TokenType.NUMBER

    def test_tabs_skipped(self):
        toks = all_tokens("\t42\t")
        assert toks[0].type == TokenType.NUMBER

    def test_newline_produces_token(self):
        toks = all_tokens("a\nb")
        types = [t.type for t in toks]
        assert TokenType.NEWLINE in types

    def test_multiple_newlines(self):
        toks = all_tokens("a\n\n\nb")
        newlines = [t for t in toks if t.type == TokenType.NEWLINE]
        assert len(newlines) == 3


# ── Comments ──────────────────────────────────────────────────────────

class TestComments:
    def test_comment_ignored(self):
        toks = all_tokens("# this is a comment\n42")
        non_eof = [t for t in toks if t.type != TokenType.EOF]
        # NEWLINE after comment + NUMBER
        assert any(t.type == TokenType.NUMBER and t.value == 42.0 for t in non_eof)

    def test_inline_comment(self):
        toks = all_tokens("42 # inline\n")
        assert toks[0].type == TokenType.NUMBER


# ── Position tracking ─────────────────────────────────────────────────

class TestPositionTracking:
    def test_column_tracked(self):
        toks = all_tokens("abc")
        assert toks[0].col == 1

    def test_line_tracked_after_newline(self):
        toks = all_tokens("a\nb")
        b_tok = next(t for t in toks if t.value == "b")
        assert b_tok.line == 2

    def test_column_resets_after_newline(self):
        toks = all_tokens("a\nbc")
        bc_tok = next(t for t in toks if t.value == "bc")
        assert bc_tok.col == 1


# ── Error cases ───────────────────────────────────────────────────────

class TestLexicalErrors:
    def test_unknown_character(self):
        with pytest.raises(LexicalError):
            all_tokens("@")

    def test_unknown_character_reports_position(self):
        try:
            all_tokens("a + @")
        except LexicalError as e:
            assert e.col > 0
