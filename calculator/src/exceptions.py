"""
Custom exception hierarchy for the interpreter.
All exceptions carry line/col for precise error reporting.
"""


class InterpreterError(Exception):
    """Base class for all interpreter errors."""
    kind = "Error"

    def __init__(self, message: str, line: int = 0, col: int = 0):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col

    def __str__(self):
        if self.line:
            return f"{self.kind} at Line {self.line}, Column {self.col}: {self.message}"
        return f"{self.kind}: {self.message}"


class LexicalError(InterpreterError):
    kind = "Lexical Error"


class SyntaxError_(InterpreterError):
    """Named with trailing underscore to avoid shadowing Python builtin."""
    kind = "Syntax Error"


class SemanticError(InterpreterError):
    kind = "Semantic Error"


class ConstraintError(InterpreterError):
    kind = "Constraint Error"


class RuntimeError_(InterpreterError):
    """Named with trailing underscore to avoid shadowing Python builtin."""
    kind = "Runtime Error"
