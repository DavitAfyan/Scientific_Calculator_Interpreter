**Project:** Scientific Calculator Interpreter  

## Overview
This project is a custom-built interpreter for a scientific calculator. It reads mathematical expressions, parses them using a custom recursive-descent parser, and evaluates the results. The interpreter supports typical arithmetic operations, compound expressions, user-defined variables, and a wide array of mathematical functions. 

Following project regulations, this interpreter is implemented **from scratch** without the use of external parsing libraries or built-in regular expression modules. It utilizes a "lazy evaluation" pull architecture, reading exactly one character at a time from the source stream to ensure optimal memory usage.

## Features
* **Custom Lexical & Syntax Analysis:** Manually implemented Lexer (Scanner) and Parser building an Abstract Syntax Tree (AST).
* **Mathematical Operations:** Supports `+`, `-`, `*`, `/`, and right-associative exponentiation `^`.
* **Standard Math Library:** Built-in support for trigonometry (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`), logarithms (`log`, `log2`, `log10`, `logb`, `ln`), roots (`sqrt`, `root`), and utility functions (`abs`, `ceil`, `floor`, `round`, `exp`).
* **Variable State Management:** Supports assigning values to variables (e.g., `radius = 5`) and storing them in a persistent symbol table across a session.
* **Granular Error Reporting:** Tracks line and column numbers natively, providing highly specific coordinates for Lexical, Syntax, Semantic, Constraint, and Runtime errors.

## Architecture Pipeline
1. **SourceReader:** Manages the input stream (file or `stdin`), reading strictly one character at a time.
2. **Scanner (Lexer):** Consumes characters from the reader via a state-machine logic to produce Tokens on demand.
3. **Parser:** Implements a recursive-descent strategy based on a left-factored EBNF grammar to build the AST and handle operation precedence.
4. **Evaluator:** Traverses the AST, executes semantic checks (e.g., undefined variables, argument counts), and computes the final floating-point values.

## Requirements & Setup
* **Python 3.x**: The main application uses only the standard library. No external dependencies are required to run the calculator.
* **Pytest**: Required *only* for running the test suite.

To install the testing requirements, open terminal/PowerShell and run:
```bash
pip install pytest

```

## Usage

Calculator can be run in interactive REPL mode, pass a file containing expressions, or pipe input directly into it.

### 1. Interactive REPL Mode

If no file is provided, the program starts an interactive session. Variables persist across the session.

```bash
python3 calculator/calculator.py

```

```text
Scientific Calculator (type 'exit' to quit)
>> a = 10
a = 10
>> b = a * sin(3.14159 / 2)
b = 10
>> exit

```

### 2. File Mode

Evaluate all statements inside a given text file sequentially.

```bash
python3 calculator/calculator.py input.txt

```

### 3. Command Line Arguments

* `input_file`: (Optional) Path to the file containing expressions.
* `--max-var-len N`: (Optional) Enforces a maximum character limit `N` on variable names (triggers a `ConstraintError` if exceeded).

**Example:**

```bash
python3 calculator/calculator.py my_expressions.txt --max-var-len 8

```

## Error Handling Examples

The interpreter provides exact coordinates for debugging.

* **Syntax Error:** `Syntax Error at Line 1, Column 10: Unexpected token '*'. Expected a number or expression.`
* **Semantic Error:** `Semantic Error at Line 1, Column 10: Function 'custom_log' is not defined.`
* **Runtime Error:** `Runtime Error at Line 1, Column 3: Division by zero.`
* **Constraint Error:** `Constraint Error at Line 1, Column 1: Variable name 'theta' exceeds maximum length of 3 characters.`

## Testing

The project includes a comprehensive test suite as well as unit tests covering component verification (Scanner, Parser, Evaluator) and integration scenarios. The tests are written using the `pytest` framework.

### How to Run Tests

Ensure you have installed `pytest` (`pip install pytest`).

**Run the entire test suite:**
This will automatically discover and run all tests.

```bash
pytest -v

```
