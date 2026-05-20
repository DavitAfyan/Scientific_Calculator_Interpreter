#!/usr/bin/env python3
"""
calculator.py - Scientific Calculator Interpreter entry point.

Usage:
    python3 calculator.py [input_file] [--max-var-len N]

    input_file      Path to a file containing expressions (optional).
                    If omitted, reads from stdin (REPL mode: one expression per line).
    --max-var-len N Maximum allowed length for variable names (optional).

Examples:
    python3 calculator.py input.txt --max-var-len 10
    echo "sin(3.14159/2)" | python3 calculator.py
    python3 calculator.py          # interactive REPL
"""

import sys
import argparse
import io

from src.source_reader import SourceReader
from src.scanner import Scanner
from src.parser import Parser
from src.evaluator import Evaluator
from src.exceptions import InterpreterError


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Scientific Calculator Interpreter (ECOTE Summer 2026)",
    )
    ap.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Path to input file. If omitted, reads from stdin.",
    )
    ap.add_argument(
        "--max-var-len",
        type=int,
        default=None,
        metavar="N",
        help="Maximum variable name length (positive integer).",
    )
    return ap


def run_stream(stream, max_var_len: int | None, output_stream=sys.stdout):
    """
    Parse and evaluate all statements from `stream`.
    Writes results to `output_stream`.
    Returns True on success, False if an error occurred.
    """
    reader = SourceReader(stream)
    scanner = Scanner(reader, max_var_len=max_var_len)
    parser = Parser(scanner)
    evaluator = Evaluator(max_var_len=max_var_len)

    try:
        statements = parser.parse_program()
        output_lines = evaluator.execute(statements)
        for line in output_lines:
            print(line, file=output_stream)
        return True
    except InterpreterError as e:
        print(str(e), file=sys.stderr)
        return False


def run_repl(max_var_len: int | None):
    """
    Interactive REPL: read one line at a time from stdin, evaluate it, and print the result.
    The evaluator is shared across lines so variables persist.
    """
    evaluator = Evaluator(max_var_len=max_var_len)
    print("Scientific Calculator (type 'exit' to quit)")

    while True:
        try:
            try:
                line = input(">> ")
            except EOFError:
                print()
                break

            if line.strip().lower() in ("exit", "quit"):
                break
            if not line.strip():
                continue

            # Wrap the single line in a stream for SourceReader
            stream = io.StringIO(line + "\n")
            reader = SourceReader(stream)
            scanner = Scanner(reader, max_var_len=max_var_len)
            parser = Parser(scanner)

            try:
                statements = parser.parse_program()
                output_lines = evaluator.execute(statements)
                for out in output_lines:
                    print(out)
            except InterpreterError as e:
                print(str(e), file=sys.stderr)

        except KeyboardInterrupt:
            print()
            break


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    max_var_len = args.max_var_len
    if max_var_len is not None and max_var_len < 1:
        print("Error: --max-var-len must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    if args.input_file is not None:
        # File mode
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                success = run_stream(f, max_var_len)
        except FileNotFoundError:
            print(f"Error: File '{args.input_file}' not found.", file=sys.stderr)
            sys.exit(1)
        sys.exit(0 if success else 1)

    elif not sys.stdin.isatty():
        # Piped / redirected stdin: process as a full file
        success = run_stream(sys.stdin, max_var_len)
        sys.exit(0 if success else 1)

    else:
        # Interactive REPL
        run_repl(max_var_len)


if __name__ == "__main__":
    main()
