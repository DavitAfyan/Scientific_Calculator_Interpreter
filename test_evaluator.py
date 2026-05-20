"""
Unit and integration tests for the Evaluator.
Covers arithmetic, variables, functions, error handling, and end-to-end scenarios.
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
from calculator.src.exceptions import SemanticError, ConstraintError, RuntimeError_


def run(text: str, max_var_len: int = None) -> list[str]:
    """Parse and evaluate text, returning the list of output lines."""
    reader = SourceReader(io.StringIO(text))
    scanner = Scanner(reader, max_var_len=max_var_len)
    parser = Parser(scanner)
    evaluator = Evaluator(max_var_len=max_var_len)
    statements = parser.parse_program()
    return evaluator.execute(statements)


def run1(text: str, max_var_len: int = None) -> str:
    """Evaluate a single expression and return its output string."""
    return run(text, max_var_len)[0]


# ── Number formatting ─────────────────────────────────────────────────

class TestFormatNumber:
    def test_integer_value(self):
        assert _format_number(2.0) == "2"

    def test_negative_integer(self):
        assert _format_number(-3.0) == "-3"

    def test_zero(self):
        assert _format_number(0.0) == "0"

    def test_float(self):
        result = _format_number(3.14)
        assert result.startswith("3.14")

    def test_trailing_zeros_stripped(self):
        assert _format_number(1.10000) == "1.1"

    def test_precision_11_places(self):
        # sin(20) formatted to at most 11 decimal places
        val = math.sin(20)
        result = _format_number(val)
        decimal_part = result.split(".")[1] if "." in result else ""
        assert len(decimal_part) <= 11


# ── Basic arithmetic ──────────────────────────────────────────────────

class TestArithmetic:
    def test_addition(self):
        assert run1("1 + 2") == "3"

    def test_subtraction(self):
        assert run1("10 - 4") == "6"

    def test_multiplication(self):
        assert run1("3 * 4") == "12"

    def test_division(self):
        assert run1("10 / 4") == "2.5"

    def test_integer_division_result(self):
        assert run1("8 / 2") == "4"

    def test_exponentiation(self):
        assert run1("2 ^ 10") == "1024"

    def test_unary_negation(self):
        assert run1("-5") == "-5"

    def test_unary_negation_precedence(self):
        # -2^2 must be -(2^2) = -4, not (-2)^2 = 4
        assert run1("-2^2") == "-4"

    def test_double_negation(self):
        assert run1("--5") == "5"

    def test_precedence_mul_over_add(self):
        assert run1("2 + 3 * 4") == "14"

    def test_precedence_parens(self):
        assert run1("(2 + 3) * 4") == "20"

    def test_right_assoc_exponentiation(self):
        # 2^3^2 = 2^(3^2) = 2^9 = 512
        assert run1("2^3^2") == "512"

    def test_mixed_precedence(self):
        # 2^3^2 - -5*2 = 512 - (-10) = 522
        assert run1("2^3^2 - -5*2") == "522"


# ── Division by zero ──────────────────────────────────────────────────

class TestDivisionByZero:
    def test_literal_zero(self):
        with pytest.raises(RuntimeError_):
            run("1 / 0")

    def test_expression_zero(self):
        with pytest.raises(RuntimeError_):
            run("10 / (5 - 5)")


# ── Variables ─────────────────────────────────────────────────────────

class TestVariables:
    def test_assign_and_read(self):
        lines = run("x = 5\nx")
        assert lines[0] == "x = 5"
        assert lines[1] == "5"

    def test_assign_expression(self):
        lines = run("y = 3 * 4 + 1")
        assert lines[0] == "y = 13"

    def test_variable_in_expression(self):
        lines = run("a = 10\na + 5")
        assert lines[1] == "15"

    def test_variable_reassignment(self):
        lines = run("b = 1\nb = b + 1")
        assert lines[1] == "b = 2"

    def test_undefined_variable(self):
        with pytest.raises(SemanticError):
            run("z + 1")

    def test_reserved_name_as_variable(self):
        with pytest.raises(SemanticError):
            run("sin = 5")

    def test_reserved_name_cos(self):
        with pytest.raises(SemanticError):
            run("cos = 1")

    def test_max_var_len_ok(self):
        lines = run("ab = 1", max_var_len=3)
        assert lines[0] == "ab = 1"

    def test_max_var_len_exceeded_assign(self):
        with pytest.raises(ConstraintError):
            run("abcd = 1", max_var_len=3)

    def test_max_var_len_exceeded_read(self):
        with pytest.raises(ConstraintError):
            # Variable name exceeds limit even when reading
            run("abcd", max_var_len=3)

    def test_multiple_variables_persist(self):
        lines = run("a = 3\nb = 4\nc = a + b")
        assert lines[2] == "c = 7"


# ── Functions ─────────────────────────────────────────────────────────

class TestFunctions:
    def test_sin(self):
        result = run1("sin(0)")
        assert result == "0"

    def test_cos(self):
        result = run1("cos(0)")
        assert result == "1"

    def test_sqrt(self):
        assert run1("sqrt(4)") == "2"
        assert run1("sqrt(9)") == "3"

    def test_abs_positive(self):
        assert run1("abs(5)") == "5"

    def test_abs_negative(self):
        assert run1("abs(-100)") == "100"

    def test_pow(self):
        assert run1("pow(2, 8)") == "256"

    def test_pow_with_abs(self):
        # pow(2, abs(-3)) = pow(2, 3) = 8
        assert run1("pow(2, abs(-3))") == "8"

    def test_root_cube(self):
        assert run1("root(-8, 3)") == "-2"

    def test_root_square(self):
        assert run1("root(9, 2)") == "3"

    def test_log_natural(self):
        result = run1("log(1)")
        assert result == "0"

    def test_ln_alias(self):
        assert run1("ln(1)") == "0"

    def test_log10(self):
        assert run1("log10(100)") == "2"

    def test_log2(self):
        assert run1("log2(8)") == "3"

    def test_logb_arbitrary_base(self):
        # logb(8, 2) = log_2(8) = 3
        assert run1("logb(8, 2)") == "3"

    def test_exp(self):
        result = run1("exp(0)")
        assert result == "1"

    def test_floor(self):
        assert run1("floor(3.9)") == "3"

    def test_ceil(self):
        assert run1("ceil(3.1)") == "4"

    def test_round(self):
        assert run1("round(3.5)") == "4"

    def test_nested_functions(self):
        # sqrt(abs(-4)) = sqrt(4) = 2
        assert run1("sqrt(abs(-4))") == "2"

    def test_unknown_function(self):
        with pytest.raises(SemanticError):
            run("custom_log(100)")

    def test_wrong_arg_count(self):
        with pytest.raises(SemanticError):
            run("sin(1, 2)")

    def test_sqrt_negative(self):
        with pytest.raises(RuntimeError_):
            run("sqrt(-1)")

    def test_log_non_positive(self):
        with pytest.raises(RuntimeError_):
            run("log(-1)")

    def test_asin_domain(self):
        with pytest.raises(RuntimeError_):
            run("asin(2)")

    def test_root_even_negative(self):
        with pytest.raises(RuntimeError_):
            run("root(-4, 2)")

    def test_atan2(self):
        # atan2(0, 1) = 0
        result = run1("atan2(0, 1)")
        assert result == "0"

    def test_hyperbolic(self):
        assert run1("sinh(0)") == "0"
        assert run1("cosh(0)") == "1"
        assert run1("tanh(0)") == "0"


# ── Integration: spec example ─────────────────────────────────────────

class TestSpecExample:
    """End-to-end test from the project specification."""

    def test_full_spec_example(self):
        text = (
            "a = sin(20)\n"
            "b = a + abs(-100)\n"
            "sqrt(4)\n"
            "b = b - pow(10,2) + 20\n"
            "root(-8, 3)\n"
        )
        lines = run(text)
        assert len(lines) == 5
        # a = sin(20)
        assert lines[0].startswith("a = ")
        a_val = float(lines[0].split("= ")[1])
        assert abs(a_val - math.sin(20)) < 1e-9
        # b = a + 100
        assert lines[1].startswith("b = ")
        b_val = float(lines[1].split("= ")[1])
        assert abs(b_val - (math.sin(20) + 100)) < 1e-9
        # sqrt(4) = 2
        assert lines[2] == "2"
        # b = b - 100 + 20
        assert lines[3].startswith("b = ")
        b2_val = float(lines[3].split("= ")[1])
        assert abs(b2_val - (b_val - 100 + 20)) < 1e-9
        # root(-8, 3) = -2
        assert lines[4] == "-2"


# ── Integration: preliminary report scenarios ─────────────────────────

class TestPreliminaryScenarios:
    def test_scenario_b_right_assoc_unary(self):
        # 2^3^2 - -5*2 = 512 - (-10) = 522
        assert run1("2^3^2 - -5*2") == "522"

    def test_scenario_c_error_on_line_3(self):
        from calculator.src.exceptions import SyntaxError_
        # Line 3 has an unclosed parenthesis: c = a + (b * sin(0.5)
        # The parser detects the mismatch when it hits the newline at the end
        # of line 3 (or start of line 4) while still expecting ')'.
        text = "a = 10\nb = 20\nc = a + (b * sin(0.5)\nd = a + b\n"
        with pytest.raises(SyntaxError_) as exc_info:
            run(text)
        # Error must reference line 3 or 4 (the newline terminates line 3,
        # so the SourceReader has already advanced to line 4 at the '\n' token).
        # Either is acceptable; what matters is that a SyntaxError_ is raised.
        assert exc_info.value.line in (3, 4)

    def test_nested_function_with_variable(self):
        text = "initial_v = 10\nstep = sin(initial_v / 2) + abs(-5)\n"
        lines = run(text, max_var_len=10)
        assert lines[0] == "initial_v = 10"
        step_val = float(lines[1].split("= ")[1])
        expected = math.sin(5) + 5
        assert abs(step_val - expected) < 1e-9

    def test_comment_does_not_affect_output(self):
        lines = run("# comment\n42")
        assert lines[0] == "42"

    def test_multiline_with_comments(self):
        lines = run("x = 3 # assign\ny = x * 2 # double\ny")
        assert lines[0] == "x = 3"
        assert lines[1] == "y = 6"
        assert lines[2] == "6"
