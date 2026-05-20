"""
Unit tests for the Parser.
Tests focus on AST structure, precedence, and associativity.
"""
import io
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from calculator.src.source_reader import SourceReader
from calculator.src.scanner import Scanner
from calculator.src.parser import Parser
from calculator.src.ast_nodes import (
    LiteralNode, VariableNode, NegateNode,
    AddNode, SubNode, MulNode, DivNode, PowNode,
    FuncCallNode, AssignNode, ExpressionStatementNode,
)
from calculator.src.exceptions import SyntaxError_


def parse(text: str):
    reader = SourceReader(io.StringIO(text))
    scanner = Scanner(reader)
    parser = Parser(scanner)
    return parser.parse_program()


# ── Basic literals and variables ──────────────────────────────────────

class TestBasicParsing:
    def test_single_number(self):
        stmts = parse("42")
        assert len(stmts) == 1
        assert isinstance(stmts[0], ExpressionStatementNode)
        assert isinstance(stmts[0].expr, LiteralNode)
        assert stmts[0].expr.value == 42.0

    def test_single_variable(self):
        stmts = parse("x")
        assert isinstance(stmts[0].expr, VariableNode)
        assert stmts[0].expr.name == "x"

    def test_assignment(self):
        stmts = parse("x = 5")
        assert isinstance(stmts[0], AssignNode)
        assert stmts[0].name == "x"
        assert isinstance(stmts[0].expr, LiteralNode)

    def test_multiple_statements(self):
        stmts = parse("a = 1\nb = 2\n3")
        assert len(stmts) == 3


# ── Precedence ────────────────────────────────────────────────────────

class TestPrecedence:
    def test_addmul_precedence(self):
        # 2 + 3 * 4 should be AddNode(2, MulNode(3, 4))
        stmts = parse("2 + 3 * 4")
        expr = stmts[0].expr
        assert isinstance(expr, AddNode)
        assert isinstance(expr.left, LiteralNode)   # 2
        assert isinstance(expr.right, MulNode)       # 3 * 4

    def test_muldiv_precedence(self):
        # 6 / 2 * 3 left-assoc → MulNode(DivNode(6,2), 3)
        stmts = parse("6 / 2 * 3")
        expr = stmts[0].expr
        assert isinstance(expr, MulNode)
        assert isinstance(expr.left, DivNode)

    def test_parens_override_precedence(self):
        # (2 + 3) * 4 → MulNode(AddNode, Literal)
        stmts = parse("(2 + 3) * 4")
        expr = stmts[0].expr
        assert isinstance(expr, MulNode)
        assert isinstance(expr.left, AddNode)

    def test_unary_minus_precedence(self):
        # -2 ^ 2 → unary applies to base: Negate(PowNode(2,2))
        stmts = parse("-2 ^ 2")
        expr = stmts[0].expr
        assert isinstance(expr, NegateNode)
        assert isinstance(expr.operand, PowNode)


# ── Associativity ─────────────────────────────────────────────────────

class TestAssociativity:
    def test_addition_left_assoc(self):
        # 1 + 2 + 3 → AddNode(AddNode(1,2), 3)
        stmts = parse("1 + 2 + 3")
        expr = stmts[0].expr
        assert isinstance(expr, AddNode)
        assert isinstance(expr.left, AddNode)

    def test_subtraction_left_assoc(self):
        stmts = parse("10 - 3 - 2")
        expr = stmts[0].expr
        assert isinstance(expr, SubNode)
        assert isinstance(expr.left, SubNode)

    def test_exponentiation_right_assoc(self):
        # 2 ^ 3 ^ 2 → PowNode(2, PowNode(3, 2))
        stmts = parse("2 ^ 3 ^ 2")
        expr = stmts[0].expr
        assert isinstance(expr, PowNode)
        assert isinstance(expr.right, PowNode)   # right-assoc


# ── Negation ──────────────────────────────────────────────────────────

class TestNegation:
    def test_simple_negation(self):
        stmts = parse("-5")
        assert isinstance(stmts[0].expr, NegateNode)

    def test_double_negation(self):
        stmts = parse("--5")
        expr = stmts[0].expr
        assert isinstance(expr, NegateNode)
        assert isinstance(expr.operand, NegateNode)

    def test_negate_expression(self):
        stmts = parse("-(2 + 3)")
        assert isinstance(stmts[0].expr, NegateNode)
        assert isinstance(stmts[0].expr.operand, AddNode)


# ── Function calls ────────────────────────────────────────────────────

class TestFunctionCalls:
    def test_no_args(self):
        # Edge case – zero-arg function would be unusual but parser should handle
        stmts = parse("f()")
        expr = stmts[0].expr
        assert isinstance(expr, FuncCallNode)
        assert expr.name == "f"
        assert expr.args == []

    def test_one_arg(self):
        stmts = parse("sin(x)")
        expr = stmts[0].expr
        assert isinstance(expr, FuncCallNode)
        assert expr.name == "sin"
        assert len(expr.args) == 1

    def test_two_args(self):
        stmts = parse("pow(2, 10)")
        expr = stmts[0].expr
        assert isinstance(expr, FuncCallNode)
        assert len(expr.args) == 2

    def test_nested_function(self):
        stmts = parse("sqrt(abs(-4))")
        outer = stmts[0].expr
        assert isinstance(outer, FuncCallNode)
        assert outer.name == "sqrt"
        assert isinstance(outer.args[0], FuncCallNode)

    def test_function_in_expression(self):
        stmts = parse("1 + sin(0)")
        expr = stmts[0].expr
        assert isinstance(expr, AddNode)
        assert isinstance(expr.right, FuncCallNode)


# ── Syntax errors ─────────────────────────────────────────────────────

class TestSyntaxErrors:
    def test_unclosed_paren(self):
        with pytest.raises(SyntaxError_):
            parse("(1 + 2")

    def test_missing_operand(self):
        with pytest.raises(SyntaxError_):
            parse("1 + * 2")

    def test_trailing_operator(self):
        with pytest.raises(SyntaxError_):
            parse("1 +")

    def test_unclosed_function_paren(self):
        with pytest.raises(SyntaxError_):
            parse("sin(x")

    def test_double_equals(self):
        with pytest.raises(SyntaxError_):
            parse("x == 5")

    def test_error_reports_line_col(self):
        try:
            parse("a\nb = 1 + * 2")
        except SyntaxError_ as e:
            assert e.line == 2
