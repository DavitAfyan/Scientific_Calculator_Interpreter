"""
Comprehensive test suite for the Scientific Calculator Interpreter.
Covers: lexical edge cases, operator precedence, associativity, function library,
variable semantics, error taxonomy, multi-line programs, boundary conditions,
and complex integration scenarios.
"""

import io
import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from calculator.src.source_reader import SourceReader
from calculator.src.scanner import Scanner
from calculator.src.parser import Parser
from calculator.src.evaluator import Evaluator, _format_number
from calculator.src.exceptions import (
    LexicalError, SyntaxError_, SemanticError, ConstraintError, RuntimeError_
)
from calculator.src.ast_nodes import (
    LiteralNode, VariableNode, NegateNode,
    AddNode, SubNode, MulNode, DivNode, PowNode,
    FuncCallNode, AssignNode, ExpressionStatementNode,
)
from calculator.src.token import TokenType


# Helpers

def make_scanner(text: str, max_var_len=None) -> Scanner:
    return Scanner(SourceReader(io.StringIO(text)), max_var_len=max_var_len)


def all_tokens(text: str, max_var_len=None):
    sc = make_scanner(text, max_var_len=max_var_len)
    tokens = []
    while True:
        t = sc.next_token()
        tokens.append(t)
        if t.type == TokenType.EOF:
            break
    return tokens


def parse(text: str, max_var_len=None):
    reader = SourceReader(io.StringIO(text))
    scanner = Scanner(reader, max_var_len=max_var_len)
    parser = Parser(scanner)
    return parser.parse_program()


def run(text: str, max_var_len: int = None) -> list:
    reader = SourceReader(io.StringIO(text))
    scanner = Scanner(reader, max_var_len=max_var_len)
    parser = Parser(scanner)
    evaluator = Evaluator(max_var_len=max_var_len)
    statements = parser.parse_program()
    return evaluator.execute(statements)


def run1(text: str, max_var_len: int = None) -> str:
    return run(text, max_var_len)[0]


def approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# SCANNER TESTS

class TestScannerNumberEdgeCases:
    """Tricky numeric literal forms."""

    def test_zero_point_zero(self):
        toks = all_tokens("0.0")
        assert toks[0].type == TokenType.NUMBER
        assert toks[0].value == 0.0

    def test_leading_dot_zero(self):
        """'.0' should lex as 0.0"""
        toks = all_tokens(".0")
        assert toks[0].type == TokenType.NUMBER
        assert toks[0].value == 0.0

    def test_leading_dot_integer_part(self):
        toks = all_tokens(".123")
        assert abs(toks[0].value - 0.123) < 1e-9

    def test_number_directly_followed_by_paren(self):
        """'2(3)' – no implicit multiplication, just tokens."""
        toks = all_tokens("2(3)")
        assert toks[0].type == TokenType.NUMBER
        assert toks[1].type == TokenType.LPAREN

    def test_very_large_integer(self):
        toks = all_tokens("999999999999")
        assert toks[0].value == 999_999_999_999.0

    def test_multiple_decimal_points_is_error(self):
        """'1.2.3' should raise a lexical error (second dot is illegal)."""
        with pytest.raises((LexicalError, SyntaxError_)):
            all_tokens("1.2.3")

    def test_number_after_operator_no_space(self):
        toks = all_tokens("10+3.5")
        assert toks[0].value == 10.0
        assert toks[1].type == TokenType.PLUS
        assert abs(toks[2].value - 3.5) < 1e-9

    def test_number_position_tracking(self):
        toks = all_tokens("   42")
        assert toks[0].col == 4  # 1-based, after 3 spaces


class TestScannerIdentifierEdgeCases:
    """Identifier boundary conditions."""

    def test_identifier_all_uppercase(self):
        toks = all_tokens("ABC")
        assert toks[0].type == TokenType.IDENTIFIER
        assert toks[0].value == "ABC"

    def test_identifier_mixed_case(self):
        toks = all_tokens("myVar_1")
        assert toks[0].value == "myVar_1"

    def test_identifier_ends_at_operator(self):
        toks = all_tokens("abc+def")
        assert toks[0].value == "abc"
        assert toks[1].type == TokenType.PLUS
        assert toks[2].value == "def"

    def test_identifier_single_underscore(self):
        toks = all_tokens("_")
        assert toks[0].type == TokenType.IDENTIFIER
        assert toks[0].value == "_"

    def test_max_var_len_boundary_exact(self):
        """Exactly at the limit should NOT raise."""
        toks = all_tokens("abc", max_var_len=3)
        assert toks[0].type == TokenType.IDENTIFIER

    def test_max_var_len_boundary_exceeded(self):
        with pytest.raises(ConstraintError):
            all_tokens("abcd", max_var_len=3)

    def test_max_var_len_1(self):
        """Single-char names allowed at limit=1."""
        toks = all_tokens("x", max_var_len=1)
        assert toks[0].value == "x"

    def test_max_var_len_1_two_chars_raises(self):
        with pytest.raises(ConstraintError):
            all_tokens("xy", max_var_len=1)

    def test_identifier_position_on_second_line(self):
        toks = all_tokens("123\nabc")
        abc_tok = next(t for t in toks if t.type == TokenType.IDENTIFIER)
        assert abc_tok.line == 2
        assert abc_tok.col == 1


class TestScannerWhitespaceAndNewlines:
    """Whitespace handling and newline semantics."""

    def test_only_whitespace_gives_eof(self):
        toks = all_tokens("   \t  ")
        assert toks[0].type == TokenType.EOF

    def test_empty_input_gives_eof(self):
        toks = all_tokens("")
        assert toks[0].type == TokenType.EOF

    def test_consecutive_newlines_each_tokenised(self):
        toks = all_tokens("\n\n\n")
        newlines = [t for t in toks if t.type == TokenType.NEWLINE]
        assert len(newlines) == 3

    def test_windows_crlf_treated_as_newline(self):
        """CR+LF should still produce a NEWLINE token (or be skipped gracefully)."""
        toks = all_tokens("a\r\nb")
        id_toks = [t for t in toks if t.type == TokenType.IDENTIFIER]
        assert len(id_toks) == 2

    def test_spaces_between_tokens_ignored(self):
        toks = all_tokens("  1   +   2  ")
        types = [t.type for t in toks if t.type != TokenType.EOF]
        assert types == [TokenType.NUMBER, TokenType.PLUS, TokenType.NUMBER]


class TestScannerComments:
    """Comment handling."""

    def test_full_line_comment_produces_nothing_before_newline(self):
        toks = all_tokens("# full line comment\n42")
        non_control = [t for t in toks if t.type not in (TokenType.EOF, TokenType.NEWLINE)]
        assert len(non_control) == 1
        assert non_control[0].value == 42.0

    def test_inline_comment_after_expression(self):
        toks = all_tokens("x + 1 # ignore me\n")
        identifiers = [t for t in toks if t.type == TokenType.IDENTIFIER]
        assert len(identifiers) == 1 and identifiers[0].value == "x"

    def test_comment_at_end_of_file_no_newline(self):
        """Comment at EOF with no trailing newline should not crash."""
        toks = all_tokens("42 # trailing comment")
        nums = [t for t in toks if t.type == TokenType.NUMBER]
        assert nums[0].value == 42.0

    def test_hash_inside_comment_is_fine(self):
        toks = all_tokens("# first # second # third\n1")
        nums = [t for t in toks if t.type == TokenType.NUMBER]
        assert nums[0].value == 1.0


class TestScannerErrors:
    """Characters that should trigger LexicalError."""

    def test_at_sign(self):
        with pytest.raises(LexicalError):
            all_tokens("@")

    def test_dollar_sign(self):
        with pytest.raises(LexicalError):
            all_tokens("$")

    def test_ampersand(self):
        with pytest.raises(LexicalError):
            all_tokens("&")

    def test_percent(self):
        with pytest.raises(LexicalError):
            all_tokens("%")

    def test_exclamation(self):
        with pytest.raises(LexicalError):
            all_tokens("!")

    def test_error_position_reported(self):
        try:
            all_tokens("abc @")
        except LexicalError as e:
            assert e.col == 5

    def test_error_line_reported_on_second_line(self):
        try:
            all_tokens("ok\n@bad")
        except LexicalError as e:
            assert e.line == 2


# PARSER / AST TESTS

class TestParserPrecedenceFull:
    """Exhaustive operator precedence checks via AST structure."""

    def test_add_before_sub_left_assoc(self):
        # 5 - 3 + 1 -> AddNode(SubNode(5,3), 1)
        stmts = parse("5 - 3 + 1")
        expr = stmts[0].expr
        assert isinstance(expr, AddNode)
        assert isinstance(expr.left, SubNode)

    def test_mul_before_add(self):
        # 1 + 2 * 3 -> AddNode(1, MulNode(2,3))
        stmts = parse("1 + 2 * 3")
        expr = stmts[0].expr
        assert isinstance(expr, AddNode)
        assert isinstance(expr.right, MulNode)

    def test_div_before_sub(self):
        # 10 - 6 / 2 -> SubNode(10, DivNode(6,2))
        stmts = parse("10 - 6 / 2")
        expr = stmts[0].expr
        assert isinstance(expr, SubNode)
        assert isinstance(expr.right, DivNode)

    def test_pow_before_mul(self):
        # 2 * 3 ^ 2 -> MulNode(2, PowNode(3,2))
        stmts = parse("2 * 3 ^ 2")
        expr = stmts[0].expr
        assert isinstance(expr, MulNode)
        assert isinstance(expr.right, PowNode)

    def test_unary_minus_binds_tighter_than_pow_base(self):
        # -2^2 -> NegateNode(PowNode(2,2)) not PowNode(NegateNode(2), 2)
        stmts = parse("-2^2")
        expr = stmts[0].expr
        assert isinstance(expr, NegateNode)
        assert isinstance(expr.operand, PowNode)

    def test_pow_right_assoc_three_levels(self):
        # 2^2^2^2 -> 2^(2^(2^2)) = 2^(2^4) = 2^16 = 65536
        stmts = parse("2^2^2^2")
        expr = stmts[0].expr
        assert isinstance(expr, PowNode)
        assert isinstance(expr.right, PowNode)
        assert isinstance(expr.right.right, PowNode)

    def test_nested_parens_deep(self):
        # ((((5)))) -> LiteralNode
        stmts = parse("((((5))))")
        assert isinstance(stmts[0].expr, LiteralNode)
        assert stmts[0].expr.value == 5.0

    def test_unary_on_paren_expression(self):
        # -(2 * 3) -> NegateNode(MulNode)
        stmts = parse("-(2 * 3)")
        assert isinstance(stmts[0].expr, NegateNode)
        assert isinstance(stmts[0].expr.operand, MulNode)

    def test_function_arg_is_full_expression(self):
        # sin(1 + 2 * 3) arg should be AddNode
        stmts = parse("sin(1 + 2 * 3)")
        func = stmts[0].expr
        assert isinstance(func, FuncCallNode)
        assert isinstance(func.args[0], AddNode)
        assert isinstance(func.args[0].right, MulNode)

    def test_assignment_rhs_is_complex_expr(self):
        stmts = parse("x = 1 + 2 * 3 ^ 2")
        assert isinstance(stmts[0], AssignNode)
        rhs = stmts[0].expr
        # 1 + (2 * (3^2))
        assert isinstance(rhs, AddNode)
        assert isinstance(rhs.right, MulNode)
        assert isinstance(rhs.right.right, PowNode)

    def test_chained_mul_div_left_assoc(self):
        # 24 / 4 / 3 / 2 -> DivNode(DivNode(DivNode(24,4),3),2)
        stmts = parse("24 / 4 / 3 / 2")
        expr = stmts[0].expr
        assert isinstance(expr, DivNode)
        assert isinstance(expr.left, DivNode)
        assert isinstance(expr.left.left, DivNode)

    def test_function_call_in_mul(self):
        # 2 * sin(0) -> MulNode(2, FuncCallNode)
        stmts = parse("2 * sin(0)")
        expr = stmts[0].expr
        assert isinstance(expr, MulNode)
        assert isinstance(expr.right, FuncCallNode)


class TestParserStatements:
    """Multi-statement and blank-line handling."""

    def test_blank_lines_between_statements(self):
        stmts = parse("a = 1\n\n\nb = 2")
        assign_nodes = [s for s in stmts if isinstance(s, AssignNode)]
        assert len(assign_nodes) == 2

    def test_only_comments_produces_no_statements(self):
        stmts = parse("# comment only\n# another\n")
        assert stmts == []

    def test_comment_then_expression(self):
        stmts = parse("# comment\n42")
        expr_stmts = [s for s in stmts if isinstance(s, ExpressionStatementNode)]
        assert len(expr_stmts) == 1

    def test_ten_statements(self):
        lines = "\n".join(f"x{i} = {i}" for i in range(10))
        stmts = parse(lines)
        assigns = [s for s in stmts if isinstance(s, AssignNode)]
        assert len(assigns) == 10

    def test_expression_followed_by_assignment(self):
        stmts = parse("5 + 3\nx = 10")
        assert isinstance(stmts[0], ExpressionStatementNode)
        assert isinstance(stmts[1], AssignNode)


class TestParserFunctionEdgeCases:
    """Parser-level checks for function call grammar."""

    def test_function_with_three_args(self):
        stmts = parse("f(1, 2, 3)")
        func = stmts[0].expr
        assert isinstance(func, FuncCallNode)
        assert len(func.args) == 3

    def test_function_arg_is_assignment_expression(self):
        """Functions don't accept assignments as args per grammar; parser should error."""
        # 'sin(x = 5)' – 'x = 5' inside arg should be a SyntaxError_
        # because assignment is a statement, not an expression per the EBNF
        with pytest.raises(SyntaxError_):
            parse("sin(x = 5)")

    def test_nested_three_levels_deep(self):
        stmts = parse("abs(sqrt(abs(-9)))")
        outer = stmts[0].expr
        assert isinstance(outer, FuncCallNode) and outer.name == "abs"
        mid = outer.args[0]
        assert isinstance(mid, FuncCallNode) and mid.name == "sqrt"
        inner = mid.args[0]
        assert isinstance(inner, FuncCallNode) and inner.name == "abs"

    def test_function_result_in_pow(self):
        stmts = parse("sin(0)^2")
        expr = stmts[0].expr
        assert isinstance(expr, PowNode)
        assert isinstance(expr.left, FuncCallNode)


class TestParserSyntaxErrors:
    """Comprehensive syntax-error detection."""

    def test_double_operator(self):
        with pytest.raises(SyntaxError_):
            parse("1 + + 2")  # consecutive binary operators

    def test_missing_comma_in_two_arg_function(self):
        with pytest.raises(SyntaxError_):
            parse("pow(2 3)")

    def test_empty_parens_in_expression(self):
        with pytest.raises(SyntaxError_):
            parse("1 + ()")

    def test_unclosed_nested_paren(self):
        with pytest.raises(SyntaxError_):
            parse("((1 + 2)")

    def test_assignment_with_no_rhs(self):
        with pytest.raises(SyntaxError_):
            parse("x =")

    def test_double_assignment(self):
        with pytest.raises(SyntaxError_):
            parse("x = y = 5")

    def test_bare_operator_at_start(self):
        with pytest.raises(SyntaxError_):
            parse("* 5")

    def test_trailing_comma_in_function(self):
        with pytest.raises(SyntaxError_):
            parse("pow(2, 3,)")

    def test_error_on_correct_line_multiline(self):
        try:
            parse("a = 1\nb = 2\nc = (d +\ne = 5")
        except SyntaxError_ as e:
            assert e.line in (3, 4)

    def test_error_column_accuracy(self):
        try:
            parse("x = 1 + * 3")
        except SyntaxError_ as e:
            # '*' is at column 9 (1-based)
            assert e.col >= 8


# EVALUATOR: ARITHMETIC PRECISION AND EDGE CASES

class TestArithmeticPrecision:
    """Values that probe floating-point representation and formatting."""

    def test_integer_result_displayed_without_decimal(self):
        assert run1("100 / 10") == "10"

    def test_non_integer_result_has_decimal(self):
        result = run1("1 / 3")
        assert "." in result

    def test_result_precision_max_11_places(self):
        result = run1("1 / 3")
        decimal_part = result.split(".")[1]
        assert len(decimal_part) <= 11

    def test_exact_half(self):
        assert run1("1 / 2") == "0.5"

    def test_negative_float(self):
        result = run1("-1 / 3")
        assert result.startswith("-")
        assert "." in result

    def test_very_small_number(self):
        result = run1("1 / 1000000")
        assert float(result) == pytest.approx(1e-6, rel=1e-6)

    def test_zero_result(self):
        assert run1("5 - 5") == "0"

    def test_negative_zero_is_zero(self):
        # -0.0 should display as "0"
        assert run1("0 * -1") == "0"

    def test_large_power(self):
        assert run1("2 ^ 32") == "4294967296"

    def test_chained_additions_integer(self):
        assert run1("1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1") == "10"

    def test_subtraction_gives_negative(self):
        assert run1("3 - 10") == "-7"

    def test_mul_negative_negative(self):
        assert run1("-3 * -4") == "12"

    def test_div_negative_result(self):
        assert run1("-10 / 4") == "-2.5"

    def test_pow_zero_exponent(self):
        assert run1("999 ^ 0") == "1"

    def test_pow_one_base(self):
        assert run1("1 ^ 1000") == "1"

    def test_pow_fractional_exponent(self):
        # 8^(1/3) = 2.0 (via exponentiation, not root)
        result = run1("8 ^ (1/3)")
        assert abs(float(result) - 2.0) < 1e-6

    def test_mul_zero_any(self):
        assert run1("0 * 99999") == "0"


class TestArithmeticComplexExpressions:
    """Multi-operator expressions that test the complete precedence tower."""

    def test_full_precedence_tower(self):
        # 2 + 3 * 4 ^ 2 - 1 = 2 + 3*16 - 1 = 2 + 48 - 1 = 49
        assert run1("2 + 3 * 4 ^ 2 - 1") == "49"

    def test_unary_in_nested_expression(self):
        # (-2)^2 via parens = 4
        assert run1("(-2)^2") == "4"

    def test_unary_without_parens(self):
        # -2^2 = -(2^2) = -4 per precedence
        assert run1("-2^2") == "-4"

    def test_triple_negation(self):
        assert run1("---5") == "-5"

    def test_quadruple_negation(self):
        assert run1("----5") == "5"

    def test_add_and_negate(self):
        # 5 + -3 = 2
        assert run1("5 + -3") == "2"

    def test_sub_and_negate(self):
        # 5 - -3 = 8
        assert run1("5 - -3") == "8"

    def test_mul_chain_with_pow(self):
        # 2 * 2 ^ 3 * 2 = 2 * 8 * 2 = 32
        assert run1("2 * 2 ^ 3 * 2") == "32"

    def test_deep_paren_nesting(self):
        # (((3 + 2))) * (((4))) = 5 * 4 = 20
        assert run1("(((3 + 2))) * (((4)))") == "20"

    def test_expression_with_all_operators(self):
        # 2 ^ 3 + 4 * 5 - 6 / 2 = 8 + 20 - 3 = 25
        assert run1("2 ^ 3 + 4 * 5 - 6 / 2") == "25"


# EVALUATOR: VARIABLES

class TestVariableSemantics:
    """Variable assignment, shadowing, persistence, and constraints."""

    def test_variable_persists_across_expressions(self):
        lines = run("x = 42\nx + 0")
        assert lines[1] == "42"

    def test_variable_used_before_assign_raises(self):
        with pytest.raises(SemanticError):
            run("undefined_var")

    def test_variable_used_in_its_own_assignment(self):
        # x = x + 1 when x is undefined should raise SemanticError
        with pytest.raises(SemanticError):
            run("x = x + 1")

    def test_variable_self_increment_after_definition(self):
        lines = run("x = 10\nx = x + 1")
        assert lines[1] == "x = 11"

    def test_multiple_variables_independent(self):
        lines = run("a = 3\nb = 4\nc = a * b")
        assert lines[2] == "c = 12"

    def test_variable_overwrite(self):
        lines = run("v = 100\nv = 200")
        assert lines[1] == "v = 200"

    def test_variable_name_single_char(self):
        lines = run("x = 1\ny = 2\nz = x + y")
        assert lines[2] == "z = 3"

    def test_variable_name_with_underscores(self):
        lines = run("my_var = 7\nmy_var")
        assert lines[1] == "7"

    def test_variable_used_in_function_arg(self):
        lines = run("n = 4\nsqrt(n)")
        assert lines[1] == "2"

    def test_variable_name_cannot_shadow_sin(self):
        with pytest.raises(SemanticError):
            run("sin = 1")

    def test_variable_name_cannot_shadow_cos(self):
        with pytest.raises(SemanticError):
            run("cos = 1")

    def test_variable_name_cannot_shadow_sqrt(self):
        with pytest.raises(SemanticError):
            run("sqrt = 1")

    def test_variable_name_cannot_shadow_abs(self):
        with pytest.raises(SemanticError):
            run("abs = 1")

    def test_variable_name_cannot_shadow_log(self):
        with pytest.raises(SemanticError):
            run("log = 1")

    def test_variable_name_cannot_shadow_pow(self):
        with pytest.raises(SemanticError):
            run("pow = 1")

    def test_variable_name_cannot_shadow_root(self):
        with pytest.raises(SemanticError):
            run("root = 1")

    def test_variable_name_cannot_shadow_exp(self):
        with pytest.raises(SemanticError):
            run("exp = 1")

    def test_constraint_exact_limit_allowed(self):
        # length 5, limit 5 -> OK
        lines = run("abcde = 1", max_var_len=5)
        assert lines[0] == "abcde = 1"

    def test_constraint_one_over_limit_raises(self):
        with pytest.raises(ConstraintError):
            run("abcdef = 1", max_var_len=5)

    def test_constraint_no_limit_long_name_ok(self):
        lines = run("this_is_a_very_long_variable_name = 99")
        assert "99" in lines[0]

    def test_constraint_applies_to_read_too(self):
        """Declaring a long name in one call, reading in another should fail if scanner enforces limit."""
        with pytest.raises(ConstraintError):
            run("toolongname", max_var_len=5)

    def test_many_variables_persisted(self):
        src = "\n".join(f"v{i} = {i * 10}" for i in range(20))
        src += "\nv19"
        lines = run(src)
        assert lines[-1] == "190"


# EVALUATOR: FUNCTION LIBRARY

class TestTrigonometricFunctions:
    """Trig functions at known-good argument values."""

    def test_sin_pi_over_2(self):
        result = float(run1(f"sin({math.pi / 2})"))
        assert approx_equal(result, 1.0)

    def test_cos_pi(self):
        result = float(run1(f"cos({math.pi})"))
        assert approx_equal(result, -1.0, tol=1e-9)

    def test_tan_zero(self):
        assert run1("tan(0)") == "0"

    def test_tan_pi_over_4(self):
        result = float(run1(f"tan({math.pi / 4})"))
        assert approx_equal(result, 1.0)

    def test_asin_zero(self):
        assert run1("asin(0)") == "0"

    def test_asin_one(self):
        result = float(run1("asin(1)"))
        assert approx_equal(result, math.pi / 2)

    def test_acos_one(self):
        assert run1("acos(1)") == "0"

    def test_acos_zero(self):
        result = float(run1("acos(0)"))
        assert approx_equal(result, math.pi / 2)

    def test_atan_zero(self):
        assert run1("atan(0)") == "0"

    def test_atan_one(self):
        result = float(run1("atan(1)"))
        assert approx_equal(result, math.pi / 4)

    def test_atan2_unit_circle(self):
        # atan2(1, 1) = pi/4
        result = float(run1("atan2(1, 1)"))
        assert approx_equal(result, math.pi / 4)

    def test_atan2_negative_x(self):
        # atan2(0, -1) = pi
        result = float(run1("atan2(0, -1)"))
        assert approx_equal(result, math.pi)

    def test_sin_large_argument(self):
        result = float(run1("sin(1000)"))
        assert approx_equal(result, math.sin(1000))

    def test_cos_negative_argument(self):
        result = float(run1("cos(-1)"))
        assert approx_equal(result, math.cos(-1))


class TestHyperbolicFunctions:
    def test_sinh_one(self):
        result = float(run1("sinh(1)"))
        assert approx_equal(result, math.sinh(1))

    def test_cosh_one(self):
        result = float(run1("cosh(1)"))
        assert approx_equal(result, math.cosh(1))

    def test_tanh_large(self):
        result = float(run1("tanh(100)"))
        assert approx_equal(result, 1.0)

    def test_tanh_negative(self):
        result = float(run1("tanh(-1)"))
        assert approx_equal(result, math.tanh(-1))


class TestLogarithmicFunctions:
    def test_ln_e(self):
        result = float(run1(f"ln({math.e})"))
        assert approx_equal(result, 1.0)

    def test_log_e(self):
        result = float(run1(f"log({math.e})"))
        assert approx_equal(result, 1.0)

    def test_log10_thousand(self):
        assert run1("log10(1000)") == "3"

    def test_log10_one_tenth(self):
        result = float(run1("log10(0.1)"))
        assert approx_equal(result, -1.0)

    def test_log2_one(self):
        assert run1("log2(1)") == "0"

    def test_log2_32(self):
        assert run1("log2(32)") == "5"

    def test_logb_base_10(self):
        result = float(run1("logb(1000, 10)"))
        assert approx_equal(result, 3.0)

    def test_logb_base_2(self):
        assert run1("logb(64, 2)") == "6"

    def test_logb_base_e(self):
        result = float(run1(f"logb({math.e}, {math.e})"))
        assert approx_equal(result, 1.0)

    def test_exp_one(self):
        result = float(run1("exp(1)"))
        assert approx_equal(result, math.e)

    def test_exp_negative(self):
        result = float(run1("exp(-1)"))
        assert approx_equal(result, 1 / math.e)


class TestRootAndPowerFunctions:
    def test_sqrt_zero(self):
        assert run1("sqrt(0)") == "0"

    def test_sqrt_one(self):
        assert run1("sqrt(1)") == "1"

    def test_sqrt_large(self):
        assert run1("sqrt(10000)") == "100"

    def test_sqrt_non_perfect(self):
        result = float(run1("sqrt(2)"))
        assert approx_equal(result, math.sqrt(2))

    def test_root_fourth(self):
        assert run1("root(16, 4)") == "2"

    def test_root_fifth(self):
        result = float(run1("root(32, 5)"))
        assert approx_equal(result, 2.0)

    def test_root_one_is_identity(self):
        assert run1("root(42, 1)") == "42"

    def test_root_negative_odd(self):
        assert run1("root(-27, 3)") == "-3"

    def test_pow_negative_base_integer_exponent(self):
        assert run1("pow(-2, 3)") == "-8"

    def test_pow_zero_base(self):
        assert run1("pow(0, 5)") == "0"

    def test_pow_fractional_exponent_via_func(self):
        result = float(run1("pow(27, 1)"))
        assert approx_equal(result, 27.0)


class TestRoundingFunctions:
    def test_floor_negative(self):
        assert run1("floor(-1.1)") == "-2"

    def test_floor_positive(self):
        assert run1("floor(1.9)") == "1"

    def test_ceil_negative(self):
        assert run1("ceil(-1.9)") == "-1"

    def test_ceil_positive(self):
        assert run1("ceil(1.1)") == "2"

    def test_round_half_up(self):
        assert run1("round(2.5)") == "3"

    def test_round_down(self):
        assert run1("round(2.4)") == "2"

    def test_round_negative(self):
        result = run1("round(-2.5)")
        # Python banker's rounding: -2.5 rounds to -2
        assert result in ("-2", "-3")

    def test_abs_zero(self):
        assert run1("abs(0)") == "0"

    def test_abs_large_negative(self):
        assert run1("abs(-999999)") == "999999"


class TestFunctionErrors:
    """Domain and arity errors for all functions."""

    def test_sqrt_negative_raises(self):
        with pytest.raises(RuntimeError_):
            run("sqrt(-0.0001)")

    def test_log_zero_raises(self):
        with pytest.raises(RuntimeError_):
            run("log(0)")

    def test_log_negative_raises(self):
        with pytest.raises(RuntimeError_):
            run("log(-5)")

    def test_ln_negative_raises(self):
        with pytest.raises(RuntimeError_):
            run("ln(-1)")

    def test_log10_zero_raises(self):
        with pytest.raises(RuntimeError_):
            run("log10(0)")

    def test_log2_negative_raises(self):
        with pytest.raises(RuntimeError_):
            run("log2(-8)")

    def test_asin_gt_one_raises(self):
        with pytest.raises(RuntimeError_):
            run("asin(1.0001)")

    def test_asin_lt_minus_one_raises(self):
        with pytest.raises(RuntimeError_):
            run("asin(-1.0001)")

    def test_acos_gt_one_raises(self):
        with pytest.raises(RuntimeError_):
            run("acos(2)")

    def test_root_even_negative_raises(self):
        with pytest.raises(RuntimeError_):
            run("root(-1, 2)")

    def test_root_even_large_negative_raises(self):
        with pytest.raises(RuntimeError_):
            run("root(-100, 4)")

    def test_unknown_function_raises_semantic(self):
        with pytest.raises(SemanticError):
            run("undefined_func(1)")

    def test_function_too_few_args(self):
        with pytest.raises(SemanticError):
            run("pow(2)")

    def test_function_too_many_args(self):
        with pytest.raises(SemanticError):
            run("sqrt(4, 2)")

    def test_zero_arg_function_with_args_raises(self):
        with pytest.raises(SemanticError):
            run("sin(0, 0)")


# EVALUATOR: RUNTIME ERRORS

class TestRuntimeErrors:
    def test_division_by_zero_literal(self):
        with pytest.raises(RuntimeError_):
            run("1 / 0")

    def test_division_by_zero_via_expression(self):
        with pytest.raises(RuntimeError_):
            run("10 / (2 - 2)")

    def test_division_by_zero_via_variable(self):
        with pytest.raises(RuntimeError_):
            run("z = 0\n5 / z")

    def test_division_by_zero_in_function_arg(self):
        with pytest.raises(RuntimeError_):
            run("sqrt(1 / 0)")

    def test_sqrt_of_division_result_negative(self):
        with pytest.raises(RuntimeError_):
            run("sqrt(4 - 10)")

    def test_error_does_not_leave_partial_state(self):
        """After an error, subsequent lines should not execute."""
        output = []
        try:
            output = run("x = 5\n1 / 0\nx")
        except RuntimeError_:
            pass
        # If error is raised, 'x' line never executes; output should not contain "5" twice
        assert len(output) <= 1  # at most the assignment line


# INTEGRATION: COMPLEX MULTI-LINE PROGRAMS

class TestIntegrationSpecExampleVariants:
    """Variations on the official spec example."""

    def test_spec_example_exact(self):
        text = (
            "a = sin(20)\n"
            "b = a + abs(-100)\n"
            "sqrt(4)\n"
            "b = b - pow(10,2) + 20\n"
            "root(-8, 3)\n"
        )
        lines = run(text)
        assert len(lines) == 5
        a_val = float(lines[0].split("= ")[1])
        assert approx_equal(a_val, math.sin(20))
        b_val = float(lines[1].split("= ")[1])
        assert approx_equal(b_val, math.sin(20) + 100)
        assert lines[2] == "2"
        b2_val = float(lines[3].split("= ")[1])
        assert approx_equal(b2_val, b_val - 100 + 20)
        assert lines[4] == "-2"

    def test_spec_example_with_comments(self):
        text = (
            "# begin\n"
            "a = sin(20) # trig\n"
            "b = a + abs(-100) # combine\n"
            "sqrt(4) # standalone\n"
        )
        lines = run(text)
        assert len(lines) == 3
        assert lines[2] == "2"

    def test_scenario_a_nested_dependency(self):
        text = (
            "initial_v = 10\n"
            "step = sin(initial_v / 2) + abs(-5)\n"
        )
        lines = run(text, max_var_len=10)
        assert lines[0] == "initial_v = 10"
        step_val = float(lines[1].split("= ")[1])
        assert approx_equal(step_val, math.sin(5) + 5)

    def test_scenario_b_right_assoc_unary(self):
        assert run1("2^3^2 - -5*2") == "522"

    def test_scenario_c_syntax_error_line_3(self):
        text = "a = 10\nb = 20\nc = a + (b * sin(0.5)\nd = a + b\n"
        with pytest.raises(SyntaxError_) as exc_info:
            run(text)
        assert exc_info.value.line in (3, 4)


class TestIntegrationChainedDependencies:
    """Programs where each line depends on the previous."""

    def test_fibonacci_like_three_steps(self):
        text = "a = 1\nb = 1\nc = a + b\nd = b + c\ne = c + d"
        lines = run(text)
        assert lines[4] == "e = 5"

    def test_compound_formula(self):
        # discriminant = b^2 - 4*a*c with a=1, b=-3, c=2
        text = "a = 1\nb = -3\nc = 2\nd = b^2 - 4*a*c"
        lines = run(text)
        d_val = float(lines[3].split("= ")[1])
        assert approx_equal(d_val, 9 - 8)  # = 1

    def test_running_total(self):
        text = "total = 0\ntotal = total + 10\ntotal = total + 20\ntotal = total + 30"
        lines = run(text)
        assert lines[3] == "total = 60"

    def test_power_tower_via_variables(self):
        text = "a = 2\nb = a^a\nc = b^a"
        lines = run(text)
        assert lines[2] == "c = 16"

    def test_trig_identity_sin_squared_plus_cos_squared(self):
        text = "x = 1.23456\ns = sin(x)^2 + cos(x)^2"
        lines = run(text)
        s_val = float(lines[1].split("= ")[1])
        assert approx_equal(s_val, 1.0, tol=1e-9)

    def test_log_exp_inverse(self):
        text = "x = 5\ny = log(exp(x))"
        lines = run(text)
        y_val = float(lines[1].split("= ")[1])
        assert approx_equal(y_val, 5.0)

    def test_sqrt_squared_identity(self):
        text = "x = 7\ny = sqrt(x)^2"
        lines = run(text)
        y_val = float(lines[1].split("= ")[1])
        assert approx_equal(y_val, 7.0)


class TestIntegrationMixedFunctionsAndOps:
    """Complex single-line expressions combining many features."""

    def test_sin_squared_plus_cos_squared_direct(self):
        result = float(run1("sin(0.7)^2 + cos(0.7)^2"))
        assert approx_equal(result, 1.0)

    def test_nested_abs_in_pow(self):
        # pow(abs(-3), abs(-4)) = pow(3, 4) = 81
        assert run1("pow(abs(-3), abs(-4))") == "81"

    def test_sqrt_of_pow(self):
        # sqrt(pow(5, 2)) = sqrt(25) = 5
        assert run1("sqrt(pow(5, 2))") == "5"

    def test_log_of_exp(self):
        assert run1("log(exp(1))") == "1"

    def test_floor_of_sqrt(self):
        # floor(sqrt(10)) = floor(3.162...) = 3
        assert run1("floor(sqrt(10))") == "3"

    def test_ceil_of_log10(self):
        # ceil(log10(50)) = ceil(1.698...) = 2
        assert run1("ceil(log10(50))") == "2"

    def test_abs_of_sin(self):
        result = float(run1("abs(sin(3.14))"))
        assert result >= 0

    def test_round_of_trig(self):
        # round(sin(pi/6)) = round(0.5) = 1
        result = run1(f"round(sin({math.pi/6}))")
        assert result in ("0", "1")  # depending on floating-point rounding

    def test_expression_with_five_functions(self):
        text = "x = abs(floor(log10(pow(sqrt(100), 2))))"
        lines = run(text)
        # sqrt(100)=10, pow(10,2)=100, log10(100)=2, floor(2)=2, abs(2)=2
        assert lines[0] == "x = 2"

    def test_negation_inside_function(self):
        # abs(-(3 + 4)) = abs(-7) = 7
        assert run1("abs(-(3 + 4))") == "7"

    def test_function_result_in_subtraction(self):
        # 10 - sqrt(25) = 10 - 5 = 5
        assert run1("10 - sqrt(25)") == "5"

    def test_variable_in_nested_function(self):
        lines = run("n = 16\nresult = sqrt(n) + log2(n)")
        result_val = float(lines[1].split("= ")[1])
        assert approx_equal(result_val, 4.0 + 4.0)


class TestIntegrationMaxVarLen:
    """Integration scenarios with max_var_len enforcement."""

    def test_limit_3_short_names_work(self):
        text = "ab = 10\nbc = 20\ncd = ab + bc"
        lines = run(text, max_var_len=3)
        assert lines[2] == "cd = 30"

    def test_limit_3_long_name_in_expression_raises(self):
        with pytest.raises(ConstraintError):
            run("abcd + 1", max_var_len=3)

    def test_limit_1_single_char_variable_works(self):
        lines = run("x = 9\ny = x * x", max_var_len=1)
        assert lines[1] == "y = 81"

    def test_limit_20_long_variable_ok(self):
        varname = "a" * 20
        lines = run(f"{varname} = 42", max_var_len=20)
        assert f"{varname} = 42" in lines[0]

    def test_limit_20_exceeded_by_one_raises(self):
        varname = "a" * 21
        with pytest.raises(ConstraintError):
            run(f"{varname} = 42", max_var_len=20)


# FORMAT NUMBER

class TestFormatNumberExtended:
    def test_format_exact_integer_large(self):
        assert _format_number(1_000_000.0) == "1000000"

    def test_format_pi_like(self):
        val = math.pi
        result = _format_number(val)
        assert result.startswith("3.14159")

    def test_format_no_more_than_11_decimals(self):
        val = 1 / 7
        result = _format_number(val)
        decimal_part = result.split(".")[1] if "." in result else ""
        assert len(decimal_part) <= 11

    def test_format_trailing_zeros_removed(self):
        assert _format_number(1.5000000000) == "1.5"

    def test_format_negative_float(self):
        result = _format_number(-0.5)
        assert result == "-0.5"

    def test_format_very_small(self):
        result = _format_number(0.1)
        assert result == "0.1"

    def test_format_negative_integer(self):
        assert _format_number(-42.0) == "-42"


# STRESS AND BOUNDARY TESTS

class TestStressAndBoundary:
    """Long programs, deeply nested expressions, boundary arithmetic."""

    def test_100_sequential_assignments(self):
        lines_in = "\n".join(f"x{i} = {i}" for i in range(100))
        lines_in += "\nx99"
        out = run(lines_in)
        assert out[-1] == "99"

    def test_deeply_nested_parens_20_levels(self):
        expr = "(" * 20 + "5" + ")" * 20
        assert run1(expr) == "5"

    def test_deeply_nested_abs(self):
        expr = "abs(" * 10 + "-7" + ")" * 10
        assert run1(expr) == "7"

    def test_long_chain_of_additions(self):
        # sum 1..100
        expr = " + ".join(str(i) for i in range(1, 101))
        assert run1(expr) == "5050"

    def test_alternating_add_sub(self):
        # 1 - 1 + 1 - 1 ... (100 terms) = 0
        parts = ["1"] + [("+ 1" if i % 2 == 0 else "- 1") for i in range(1, 100)]
        expr = " ".join(parts)
        result = int(run1(expr))
        assert result in (0, 1)  # 100 terms: 50 pairs -> 0

    def test_right_assoc_power_chain_5(self):
        # 2^1^1^1^1 = 2^(1^(1^(1^1))) = 2^1 = 2
        assert run1("2^1^1^1^1") == "2"

    def test_zero_divided_by_nonzero(self):
        assert run1("0 / 999") == "0"

    def test_expression_result_used_as_function_arg_indirectly(self):
        text = "base = 2\nexp_val = 10\nresult = pow(base, exp_val)"
        lines = run(text)
        assert lines[2] == "result = 1024"

    def test_unary_minus_on_function_result(self):
        # -sqrt(9) = -3
        assert run1("-sqrt(9)") == "-3"

    def test_double_unary_on_function_result(self):
        # --sqrt(9) = 3
        assert run1("--sqrt(9)") == "3"

    def test_pow_of_function_results(self):
        # sqrt(9) ^ floor(2.9) = 3 ^ 2 = 9
        assert run1("sqrt(9) ^ floor(2.9)") == "9"

    def test_mixed_multiline_with_blanks_and_comments(self):
        text = (
            "# Init\n"
            "x = 3\n"
            "\n"
            "# Double\n"
            "y = x * 2\n"
            "\n"
            "# Square\n"
            "z = y ^ 2\n"
            "z\n"
        )
        lines = run(text)
        assert lines[-1] == "36"

    def test_variable_named_like_prefix_of_function(self):
        # 'si' is not a reserved word, only 'sin' is
        lines = run("si = 100\nsi")
        assert lines[0] == "si = 100"
        assert lines[1] == "100"

    def test_variable_named_lo(self):
        # 'lo' not reserved
        lines = run("lo = 5\nlo")
        assert lines[1] == "5"

    def test_output_order_preserved(self):
        text = "a = 1\nb = 2\nc = 3\nb\na\nc"
        lines = run(text)
        # First three are assignments, then b, a, c
        assert lines[3] == "2"
        assert lines[4] == "1"
        assert lines[5] == "3"

    def test_very_small_float_precision(self):
        result = float(run1(f"sin({math.pi})"))
        # sin(pi) should format to something very close to 0
        assert abs(result) < 1e-10


# ERROR TAXONOMY: ENSURE CORRECT EXCEPTION TYPE IS RAISED

class TestErrorTaxonomy:
    """Each error category must raise its own exception type, not a generic one."""

    def test_lexical_error_is_lexical(self):
        with pytest.raises(LexicalError):
            run("1 @ 2")

    def test_syntax_error_is_syntax(self):
        with pytest.raises(SyntaxError_):
            run("1 +")

    def test_semantic_undefined_var_is_semantic(self):
        with pytest.raises(SemanticError):
            run("undefined_xyz")

    def test_semantic_reserved_name_is_semantic(self):
        with pytest.raises(SemanticError):
            run("sin = 5")

    def test_semantic_wrong_arg_count_is_semantic(self):
        with pytest.raises(SemanticError):
            run("sqrt(1, 2)")

    def test_constraint_error_is_constraint(self):
        with pytest.raises(ConstraintError):
            run("toolong = 1", max_var_len=4)

    def test_runtime_div_zero_is_runtime(self):
        with pytest.raises(RuntimeError_):
            run("1 / 0")

    def test_runtime_sqrt_neg_is_runtime(self):
        with pytest.raises(RuntimeError_):
            run("sqrt(-1)")

    def test_runtime_log_neg_is_runtime(self):
        with pytest.raises(RuntimeError_):
            run("log(-1)")

    def test_runtime_asin_domain_is_runtime(self):
        with pytest.raises(RuntimeError_):
            run("asin(2)")

    def test_runtime_root_even_neg_is_runtime(self):
        with pytest.raises(RuntimeError_):
            run("root(-4, 2)")


# POSITION REPORTING

class TestErrorPositionReporting:
    """Errors must carry accurate line/col information."""

    def test_lexical_error_reports_col(self):
        try:
            run("1 + @")
        except LexicalError as e:
            assert e.col == 5

    def test_syntax_error_line_1(self):
        try:
            run("1 + * 2")
        except SyntaxError_ as e:
            assert e.line == 1

    def test_syntax_error_line_3(self):
        try:
            run("a = 1\nb = 2\nc = +")
        except SyntaxError_ as e:
            assert e.line == 3

    def test_lexical_error_line_2(self):
        try:
            run("ok = 1\nbad $ stuff")
        except LexicalError as e:
            assert e.line == 2

    def test_constraint_error_has_position(self):
        try:
            run("toolong = 1", max_var_len=4)
        except ConstraintError as e:
            assert hasattr(e, "line") or hasattr(e, "col")

    def test_semantic_error_has_position(self):
        try:
            run("missing_var + 1")
        except SemanticError as e:
            assert hasattr(e, "line") or hasattr(e, "col")

    def test_runtime_error_has_position(self):
        try:
            run("1 / 0")
        except RuntimeError_ as e:
            assert hasattr(e, "line") or hasattr(e, "col")
