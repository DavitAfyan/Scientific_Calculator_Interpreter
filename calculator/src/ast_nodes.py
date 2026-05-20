"""
Abstract Syntax Tree node hierarchy.
Each grammar production maps to one or more node types.
"""


class ASTNode:
    """Base class for all AST nodes."""
    pass


# Leaf nodes 

class LiteralNode(ASTNode):
    """A numeric constant, e.g. 3.14"""
    def __init__(self, value: float):
        self.value = value

    def __repr__(self):
        return f"Literal({self.value})"


class VariableNode(ASTNode):
    """A variable reference, e.g. x"""
    def __init__(self, name: str, line: int, col: int):
        self.name = name
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Var({self.name})"


# Unary nodes 

class NegateNode(ASTNode):
    """Unary negation: -expr"""
    def __init__(self, operand: ASTNode):
        self.operand = operand

    def __repr__(self):
        return f"Negate({self.operand})"


# Binary operation nodes 

class BinOpNode(ASTNode):
    """Abstract base for binary operations."""
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.__class__.__name__}({self.left}, {self.right})"


class AddNode(BinOpNode):
    pass

class SubNode(BinOpNode):
    pass

class MulNode(BinOpNode):
    pass

class DivNode(BinOpNode):
    line: int = 0
    col: int = 0

    def __init__(self, left, right, line=0, col=0):
        super().__init__(left, right)
        self.line = line
        self.col = col

class PowNode(BinOpNode):
    pass


# Function call node 

class FuncCallNode(ASTNode):
    """A function call: name(arg1, arg2, ...)"""
    def __init__(self, name: str, args: list, line: int, col: int):
        self.name = name
        self.args = args          # list of ASTNode
        self.line = line
        self.col = col

    def __repr__(self):
        return f"FuncCall({self.name}, {self.args})"


# Statement nodes 

class AssignNode(ASTNode):
    """Variable assignment: name = expr"""
    def __init__(self, name: str, expr: ASTNode, line: int, col: int):
        self.name = name
        self.expr = expr
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Assign({self.name}, {self.expr})"


class ExpressionStatementNode(ASTNode):
    """A bare expression used as a statement (its value is printed)."""
    def __init__(self, expr: ASTNode):
        self.expr = expr

    def __repr__(self):
        return f"ExprStmt({self.expr})"
