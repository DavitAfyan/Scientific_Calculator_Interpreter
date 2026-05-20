"""
Evaluator: walks the AST and computes numeric results.
All actual math is delegated to Python's math library (via functions.py).
Semantic checks (reserved names, variable existence, arg counts) happen here.
"""

from .ast_nodes import (
    LiteralNode, VariableNode, NegateNode,
    AddNode, SubNode, MulNode, DivNode, PowNode,
    FuncCallNode, AssignNode, ExpressionStatementNode,
)
from .functions import FUNCTION_REGISTRY, RESERVED_NAMES
from .exceptions import SemanticError, ConstraintError, RuntimeError_


def _format_number(value: float) -> str:
    """
    Format a float for output:
    - If the value is a whole number, show no decimal places (e.g. 2 not 2.0)
    - Otherwise show up to 11 significant digits, stripping trailing zeros
    """
    if value == value:  # Note for NaN: NaN != NaN, so this will be False for NaN
        try:
            if value == int(value):
                return str(int(value))
        except (OverflowError, ValueError):
            pass
    # Up to 11 decimal places, strip trailing zeros
    formatted = f"{value:.11f}".rstrip("0").rstrip(".")
    return formatted


class Evaluator:
    def __init__(self, max_var_len: int = None):
        self._symbol_table: dict[str, float] = {}
        self._max_var_len = max_var_len

    # Public interface

    def execute(self, statements: list) -> list[str]:
        """
        Execute a list of statement AST nodes.
        Returns a list of output lines (one per statement that produces output).
        Raises InterpreterError subclasses on any error.
        """
        output_lines = []
        for stmt in statements:
            result = self._eval_statement(stmt)
            if result is not None:
                output_lines.append(result)
        return output_lines

    # Statement dispatch

    def _eval_statement(self, node) -> str | None:
        if isinstance(node, AssignNode):
            return self._eval_assign(node)
        if isinstance(node, ExpressionStatementNode):
            value = self._eval_expr(node.expr)
            return _format_number(value)
        raise RuntimeError_(f"Unknown statement node: {type(node)}")

    def _eval_assign(self, node: AssignNode) -> str:
        name = node.name

        # Check not a reserved function name
        if name in RESERVED_NAMES:
            raise SemanticError(
                f"'{name}' is a reserved function name and cannot be used as a variable",
                node.line, node.col,
            )

        # Check variable name length constraint
        if self._max_var_len is not None and len(name) > self._max_var_len:
            raise ConstraintError(
                f"Variable name '{name}' exceeds maximum length of {self._max_var_len} characters",
                node.line, node.col,
            )

        value = self._eval_expr(node.expr)
        self._symbol_table[name] = value
        return f"{name} = {_format_number(value)}"

    # Expression dispatch

    def _eval_expr(self, node) -> float:
        if isinstance(node, LiteralNode):
            return node.value

        if isinstance(node, VariableNode):
            return self._eval_variable(node)

        if isinstance(node, NegateNode):
            return -self._eval_expr(node.operand)

        if isinstance(node, AddNode):
            return self._eval_expr(node.left) + self._eval_expr(node.right)

        if isinstance(node, SubNode):
            return self._eval_expr(node.left) - self._eval_expr(node.right)

        if isinstance(node, MulNode):
            return self._eval_expr(node.left) * self._eval_expr(node.right)

        if isinstance(node, DivNode):
            return self._eval_div(node)

        if isinstance(node, PowNode):
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            try:
                return left ** right
            except (OverflowError, ZeroDivisionError, ValueError) as e:
                raise RuntimeError_(f"Math error in exponentiation ({left} ^ {right}): {e}")

        if isinstance(node, FuncCallNode):
            return self._eval_func(node)

        raise RuntimeError_(f"Unknown expression node: {type(node)}")

    def _eval_variable(self, node: VariableNode) -> float:
        # Enforce length constraint on variable reads too
        if self._max_var_len is not None and len(node.name) > self._max_var_len:
            raise ConstraintError(
                f"Variable name '{node.name}' exceeds maximum length of {self._max_var_len} characters",
                node.line, node.col,
            )
        if node.name not in self._symbol_table:
            raise SemanticError(
                f"Undefined variable '{node.name}'",
                node.line, node.col,
            )
        return self._symbol_table[node.name]

    def _eval_div(self, node: DivNode) -> float:
        left = self._eval_expr(node.left)
        right = self._eval_expr(node.right)
        if right == 0:
            raise RuntimeError_(
                "Division by zero",
                node.line, node.col,
            )
        return left / right

    def _eval_func(self, node: FuncCallNode) -> float:
        name = node.name

        if name not in FUNCTION_REGISTRY:
            raise SemanticError(
                f"Function '{name}' is not defined",
                node.line, node.col,
            )

        expected_argc, fn = FUNCTION_REGISTRY[name]
        actual_argc = len(node.args)

        if actual_argc != expected_argc:
            raise SemanticError(
                f"Function '{name}' expects {expected_argc} argument(s), got {actual_argc}",
                node.line, node.col,
            )

        evaluated_args = [self._eval_expr(arg) for arg in node.args]

        try:
            result = fn(*evaluated_args)
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            raise RuntimeError_(
                f"Math error in '{name}({', '.join(str(a) for a in evaluated_args)})': {e}",
                node.line, node.col,
            )

        return float(result)
