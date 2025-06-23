@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two integers."""
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide two integers."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b