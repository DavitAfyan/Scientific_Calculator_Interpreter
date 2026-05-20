"""
Function registry: maps function names to (arg_count, callable).
All actual math is delegated to Python's math library.
"""

import math


def _root(x, n):
    """n-th root of x, supporting odd roots of negative numbers."""
    if x < 0:
        if n % 2 == 1:
            return -((-x) ** (1.0 / n))
        else:
            raise ValueError(f"Cannot take even root of negative number {x}")
    return x ** (1.0 / n)


def _log(x):
    """Natural logarithm."""
    if x <= 0:
        raise ValueError(f"log argument must be positive, got {x}")
    return math.log(x)


def _log2(x):
    if x <= 0:
        raise ValueError(f"log2 argument must be positive, got {x}")
    return math.log2(x)


def _log10(x):
    if x <= 0:
        raise ValueError(f"log10 argument must be positive, got {x}")
    return math.log10(x)


def _sqrt(x):
    if x < 0:
        raise ValueError(f"Cannot take square root of negative number {x}")
    return math.sqrt(x)


def _asin(x):
    if not (-1 <= x <= 1):
        raise ValueError(f"asin argument must be in [-1, 1], got {x}")
    return math.asin(x)


def _acos(x):
    if not (-1 <= x <= 1):
        raise ValueError(f"acos argument must be in [-1, 1], got {x}")
    return math.acos(x)


def _tan(x):
    return math.tan(x)


def _abs(x):
    return abs(x)


def _pow(base, exp):
    return math.pow(base, exp)


def _ceil(x):
    return float(math.ceil(x))


def _floor(x):
    return float(math.floor(x))


def _round_fn(x):
    # Round half-up (standard mathematical rounding)
    import math as _math
    return float(_math.floor(x + 0.5))


def _logb(x, base):
    """Logarithm of x in a given base."""
    if x <= 0:
        raise ValueError(f"log argument must be positive, got {x}")
    if base <= 0 or base == 1:
        raise ValueError(f"log base must be positive and not 1, got {base}")
    return math.log(x, base)


# Registry: name -> (arg_count, callable)
FUNCTION_REGISTRY = {
    # Trigonometric
    "sin":   (1, math.sin),
    "cos":   (1, math.cos),
    "tan":   (1, _tan),
    "asin":  (1, _asin),
    "acos":  (1, _acos),
    "atan":  (1, math.atan),
    "atan2": (2, math.atan2),

    # Roots & powers
    "sqrt":  (1, _sqrt),
    "root":  (2, _root),
    "pow":   (2, _pow),
    "exp":   (1, math.exp),

    # Logarithms
    "log":   (1, _log),
    "log2":  (1, _log2),
    "ln":    (1, _log),
    "log10": (1, _log10),
    "logb":  (2, _logb),   # log(x, base) with arbitrary base

    # Absolute value & rounding
    "abs":   (1, _abs),
    "ceil":  (1, _ceil),
    "floor": (1, _floor),
    "round": (1, _round_fn),

    # Hyperbolic
    "sinh":  (1, math.sinh),
    "cosh":  (1, math.cosh),
    "tanh":  (1, math.tanh),
}

# Set of reserved names that cannot be used as variables
RESERVED_NAMES = frozenset(FUNCTION_REGISTRY.keys())