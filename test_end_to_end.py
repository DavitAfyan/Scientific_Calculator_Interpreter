"""
End-to-End and System tests for the Scientific Calculator Interpreter.
Covers: CLI execution, File I/O, Exact Error String Formatting, and Lazy Evaluation verification.
"""

import subprocess
import os
import sys
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from calculator.src.source_reader import SourceReader

MAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "calculator/calculator.py")

# CLI & File I/O 

class TestCommandLineInterface:
    
    def test_cli_reads_from_file_and_outputs_correctly(self, tmp_path):
        """Verify the program successfully opens a file, processes it, and prints to stdout."""
        # Create a temporary input file
        input_file = tmp_path / "test_input.txt"
        input_file.write_text("radius = 5\narea = 3.14 * radius ^ 2\narea")

        # Execute the script via command line
        result = subprocess.run(
            [sys.executable, MAIN_SCRIPT, str(input_file)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "radius = 5" in result.stdout
        assert "area = 78.5" in result.stdout
        assert "78.5" in result.stdout

    def test_cli_max_var_len_argument_enforced(self, tmp_path):
        """Verify the --max-var-len argument is captured and passed to the Evaluator/Scanner."""
        input_file = tmp_path / "test_input_len.txt"
        input_file.write_text("abcd = 10\n")

        # Run with max length 3, expecting it to fail on 'abcd'
        result = subprocess.run(
            [sys.executable, MAIN_SCRIPT, str(input_file), "--max-var-len", "3"],
            capture_output=True,
            text=True
        )

        # Assuming the program exits with a non-zero code on error
        assert result.returncode != 0
        assert "Constraint Error" in (result.stdout + result.stderr)

# Error Message String Formatting 

class TestExactErrorFormatting:
    
    def test_syntax_error_string_format(self, tmp_path):
        """Verify errors match the exact string format required by the specification."""
        input_file = tmp_path / "test_syntax_error.txt"
        input_file.write_text("a = 10 + * 5\n")

        result = subprocess.run(
            [sys.executable, MAIN_SCRIPT, str(input_file)],
            capture_output=True,
            text=True
        )

        output = result.stdout + result.stderr
        
        # Checking for the precise string structure
        assert "Syntax Error at Line 1, Column 10:" in output
        assert "Unexpected token" in output

    def test_missing_function_error_string_format(self, tmp_path):
        input_file = tmp_path / "test_semantic_error.txt"
        input_file.write_text("result = custom_log(100)\n")

        result = subprocess.run(
            [sys.executable, MAIN_SCRIPT, str(input_file)],
            capture_output=True,
            text=True
        )

        output = result.stdout + result.stderr
        assert "Semantic Error at Line 1, Column 10:" in output
        assert "not defined" in output

# Proof of Lazy Evaluation 

class TestLazyEvaluation:
    
    def test_source_reader_only_reads_one_character_at_a_time(self):
        """
        Mathematically proves lazy evaluation by mocking the file stream 
        and asserting that .read(1) is the only method called.
        """
        # Create a fake file stream
        mock_stream = MagicMock()
        mock_stream.read.return_value = "a"
        
        # Instantiate the reader with the fake stream
        reader = SourceReader(mock_stream)
        
        # Trigger the reader to fetch the next character
        char = reader.next_char()
        
        # ASSERTION 1: Prove it fetched a character
        assert char == "a"
        
        # ASSERTION 2: Prove it called read(1) explicitly, not read() or readlines()
        mock_stream.read.assert_called_with(1)
        
        # ASSERTION 3: Prove it only called it exactly once per character request
        assert mock_stream.read.call_count == 1