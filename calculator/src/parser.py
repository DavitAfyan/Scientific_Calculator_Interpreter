"""
Parser: recursive-descent parser implementing the EBNF grammar.
Pulls tokens from the Scanner one at a time and builds an AST.

Grammar (EBNF):
    program    = { statement } ;
    statement  = assignment | expr_stmt ;
    assignment = IDENTIFIER "=" expression NEWLINE/EOF ;
    expr_stmt  = expression NEWLINE/EOF ;
    expression = term { ( "+" | "-" ) term } ;
    term       = factor { ( "*" | "/" ) factor } ;
    factor     = unary [ "^" factor ] ;          (right-associative)
    unary      = "-" unary | power ;             (unary minus has LOWER precedence than ^)
    power      = primary [ "^" factor ] ;        (right-associative, used inside unary)
    primary    = NUMBER
               | "(" expression ")"
               | IDENTIFIER [ "(" arg_list ")" ] ;
    arg_list   = expression { "," expression } ;

Precedence (lowest to highest):
    +  -        (left-associative)
    *  /        (left-associative)
    ^           (right-associative)
    unary -     (right-associative)
    primary
"""

from .scanner import Scanner
from .token import Token, TokenType
from .ast_nodes import (
    LiteralNode, VariableNode, NegateNode,
    AddNode, SubNode, MulNode, DivNode, PowNode,
    FuncCallNode, AssignNode, ExpressionStatementNode,
)
from .exceptions import SyntaxError_


class Parser:
    def __init__(self, scanner: Scanner):
        self._scanner = scanner
        self._current: Token = None
        self._advance()                # load first token

    # Token management

    def _advance(self):
        """Consume current token and load the next one, skipping blank newlines."""
        self._current = self._scanner.next_token()

    def _peek(self) -> Token:
        return self._current

    def _expect(self, ttype: TokenType) -> Token:
        """Consume and return the current token if it matches; raise otherwise."""
        tok = self._current
        if tok.type != ttype:
            raise SyntaxError_(
                f"Expected '{ttype.name}' but got '{tok.value!r}'",
                tok.line, tok.col,
            )
        self._advance()
        return tok

    def _skip_newlines(self):
        while self._current.type == TokenType.NEWLINE:
            self._advance()

    # Top-level

    def parse_program(self):
        """Parse all statements until EOF. Returns list of AST nodes."""
        statements = []
        self._skip_newlines()
        while self._current.type != TokenType.EOF:
            stmt = self._parse_statement()
            if stmt is not None:
                statements.append(stmt)
            self._skip_newlines()
        return statements

    # Statements

    def _parse_statement(self):
        """
        statement = assignment | expr_stmt
        We need one token of look-ahead to distinguish:
          IDENTIFIER "=" ...   -> assignment
          anything else        -> expression statement
        """
        tok = self._peek()

        # Possible assignment: IDENTIFIER followed by EQUALS
        if tok.type == TokenType.IDENTIFIER:
            # peek two tokens ahead without a scanner lookahead
            # parse the identifier as part of primary, then check if the next token is EQUALS
            # save position info and handle it here
            name = tok.value
            name_line = tok.line
            name_col = tok.col
            self._advance()                        # consume IDENTIFIER

            if self._peek().type == TokenType.EQUALS:
                self._advance()                    # consume "="
                expr = self._parse_expression()
                self._expect_end_of_statement()
                return AssignNode(name, expr, name_line, name_col)
            else:
                # Not an assignment: build a VariableNode and continue parsing
                # the rest of the expression with it as the left-most primary.
                left = self._finish_primary(VariableNode(name, name_line, name_col))
                left = self._finish_factor(left)
                left = self._finish_term(left)
                left = self._finish_expression(left)
                self._expect_end_of_statement()
                return ExpressionStatementNode(left)

        # Pure expression statement
        expr = self._parse_expression()
        self._expect_end_of_statement()
        return ExpressionStatementNode(expr)

    def _expect_end_of_statement(self):
        """A statement ends at NEWLINE or EOF."""
        if self._current.type in (TokenType.NEWLINE, TokenType.EOF):
            if self._current.type == TokenType.NEWLINE:
                self._advance()
        else:
            tok = self._current
            raise SyntaxError_(
                f"Expected end of statement but got '{tok.value!r}'",
                tok.line, tok.col,
            )

    # Expression grammar (precedence climbing via recursive descent)

    def _parse_expression(self):
        left = self._parse_term()
        return self._finish_expression(left)

    def _finish_expression(self, left):
        while self._peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = self._peek().type
            self._advance()
            right = self._parse_term()
            if op == TokenType.PLUS:
                left = AddNode(left, right)
            else:
                left = SubNode(left, right)
        return left

    def _parse_term(self):
        left = self._parse_factor()
        return self._finish_term(left)

    def _finish_term(self, left):
        while self._peek().type in (TokenType.STAR, TokenType.SLASH):
            op = self._peek().type
            op_tok = self._peek()
            self._advance()
            right = self._parse_factor()
            if op == TokenType.STAR:
                left = MulNode(left, right)
            else:
                left = DivNode(left, right, op_tok.line, op_tok.col)
        return left

    def _parse_factor(self):
        # factor = unary [ "^" factor ]   (right-associative)
        # Unary negation sits above factor so that -2^2 = -(2^2), not (-2)^2.
        base = self._parse_unary()
        return self._finish_factor(base)

    def _finish_factor(self, base):
        # Right-associative exponentiation
        if self._peek().type == TokenType.CARET:
            self._advance()
            exp = self._parse_factor()   # recursive for right-assoc
            return PowNode(base, exp)
        return base

    def _parse_unary(self):
        # unary = "-" unary | power
        # power = primary [ "^" factor ]
        # By calling _parse_power (not _parse_factor) from NegateNode's operand,
        # we ensure ^ binds more tightly than unary minus:
        # -2^2  ->  NegateNode(PowNode(2, 2))  ->  -(4) = -4
        if self._peek().type == TokenType.MINUS:
            self._advance()
            operand = self._parse_unary()
            return NegateNode(operand)
        return self._parse_power()

    def _parse_power(self):
        """Parse a primary optionally followed by '^' (right-assoc exponentiation)."""
        base = self._parse_primary()
        if self._peek().type == TokenType.CARET:
            self._advance()
            exp = self._parse_factor()   # factor for right-assoc, unary applies to exp
            return PowNode(base, exp)
        return base

    def _parse_primary(self):
        tok = self._peek()

        # Number literal
        if tok.type == TokenType.NUMBER:
            self._advance()
            return LiteralNode(tok.value)

        # Parenthesised expression
        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            close = self._current
            if close.type != TokenType.RPAREN:
                raise SyntaxError_(
                    f"Expected ')' but got '{close.value!r}'",
                    close.line, close.col,
                )
            self._advance()
            return expr

        # Identifier: variable or function call
        if tok.type == TokenType.IDENTIFIER:
            name = tok.value
            self._advance()
            node = VariableNode(name, tok.line, tok.col)
            return self._finish_primary(node)

        raise SyntaxError_(
            f"Unexpected token '{tok.value!r}': expected a number, '(' or identifier",
            tok.line, tok.col,
        )

    def _finish_primary(self, node: VariableNode):
        """
        After consuming an IDENTIFIER, check if it's a function call.
        If the next token is '(', treat it as FuncCallNode.
        """
        if self._peek().type == TokenType.LPAREN:
            self._advance()            # consume '('
            args = []
            if self._peek().type != TokenType.RPAREN:
                args.append(self._parse_expression())
                while self._peek().type == TokenType.COMMA:
                    self._advance()    # consume ','
                    args.append(self._parse_expression())
            close = self._current
            if close.type != TokenType.RPAREN:
                raise SyntaxError_(
                    f"Expected ')' but got '{close.value!r}'",
                    close.line, close.col,
                )
            self._advance()            # consume ')'
            return FuncCallNode(node.name, args, node.line, node.col)
        return node
