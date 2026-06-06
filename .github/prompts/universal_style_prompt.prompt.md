# Python Code Style Guide

Follow these conventions when writing or editing Python code in this workspace.

## General Principles

- Write **readable, maintainable, self-documenting code** — code is read far more than it is written.
- Prefer **explicit over implicit** — avoid clever one-liners that sacrifice clarity.
- Follow the **principle of least surprise** — behavior should match what a reasonable reader would expect.
- Keep functions and methods **short and focused** — each should do one thing well. If a function exceeds ~40 lines, consider breaking it up.
- Use **descriptive names** — avoid single-letter names except for simple loop counters (`i`, `j`) or common math conventions (`x`, `y`, `n`).

## Naming Conventions (PEP 8)

| Element | Convention | Example |
|---|---|---|
| Variables & functions | `snake_case` | `user_name`, `calculate_total()` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Classes | `PascalCase` | `DataProcessor`, `HttpClient` |
| Private attributes | `_leading_underscore` | `_internal_state`, `_validate()` |
| Module names | `snake_case`, short | `data_utils.py`, `file_handler.py` |
| Boolean variables | `is_`, `has_`, `can_` prefix | `is_active`, `has_permission` |

## Type Hints

- Add **type hints** to all function signatures (parameters and return types).
- Use `Optional[T]` for values that can be `None`, not `T | None` (for Python 3.9 compatibility).
- Use `list[T]`, `dict[K, V]`, `set[T]` (lowercase) for Python 3.9+ generics.
- Use `... -> None` for functions that don't return a value.
- Prefer **type aliases** for complex types.

```python
# Good
def fetch_user(user_id: str, include_deleted: bool = False) -> dict[str, Any]:
    ...

def process_items(items: list[str]) -> Optional[str]:
    ...

# Avoid
def fetch_user(user_id, include_deleted=False):
    ...
```

## Docstrings

- Use **Google-style docstrings** for all public functions, classes, and modules.
- Include `Args`, `Returns`, and `Raises` sections when applicable.
- Write docstrings as **imperive commands** ("Return the total" not "Returns the total").

```python
def calculate_discount(price: float, discount_percent: float) -> float:
    """Apply a percentage discount to a price.

    Args:
        price: The original price.
        discount_percent: The discount percentage (0-100).

    Returns:
        The discounted price.

    Raises:
        ValueError: If discount_percent is negative or exceeds 100.
    """
    ...
```

## Code Formatting

- **Indentation**: 4 spaces per level (no tabs).
- **Line length**: Target 88 characters (Black default), hard limit at 120 characters.
- **String quotes**: Prefer **double quotes** (`"`) for consistency, single quotes (`'`) only for strings containing double quotes.
- **Trailing commas**: Use in multi-line collections and function parameter lists.

```python
# Good — trailing comma on multi-line
TOOLS = [
    "calculator",
    "weather",
    "search",
]

# Good — multi-line function call
result = agent.invoke(
    {"messages": [("user", query)]},
    config=config,
)
```

## Imports

- Group imports in this order, separated by blank lines:
  1. **Standard library** (`os`, `time`, `logging`, `datetime`)
  2. **Third-party** (`langchain`, `openai`, `dotenv`)
  3. **Local/application** (your own modules)
- Use **absolute imports** over relative imports when possible.
- Avoid **wildcard imports** (`from module import *`).
- Keep imports **sorted alphabetically** within each group.

```python
# Good
import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from my_app.utils import format_output
```

## Error Handling

- Be **specific** with exceptions — catch the most precise exception type possible, never bare `except:`.
- Use **custom exception classes** for application-specific errors.
- Always include a **descriptive error message** that explains what went wrong and why.
- Use `logging` for error reporting, not just `print()`.

```python
# Good
except openai.RateLimitError as e:
    logger.warning("HTTP 429 rate limit hit: %s", e)
    time.sleep(wait_time)

# Avoid
except Exception:
    pass
```

## Logging

- Use Python's `logging` module instead of `print()` for production code.
- Configure logging with **timestamps and log levels**.
- Use appropriate log levels: `DEBUG` (diagnostics), `INFO` (progress), `WARNING` (unexpected but handled), `ERROR` (failures).
- Use **lazy string formatting** with `%s` in logging calls, not f-strings.

```python
# Good
logger.info("Query completed in %.3fs with %d tool call(s): %s", duration, count, query)
logger.warning("Slow query detected: %s", query)

# Avoid
logger.info(f"Query completed in {duration}s")  # f-string evaluates even if log level is disabled
print("Query completed")  # no level control
```

## Data Structures & Patterns

- Prefer **dataclasses** or **Pydantic models** over raw dictionaries for structured data.
- Use **list comprehensions** for simple transformations; use full `for` loops for complex logic.
- Use **f-strings** for string formatting (not `%` or `.format()`).
- Use **enums** for fixed sets of values.
- Prefer **`pathlib.Path`** over `os.path` for file path operations.

```python
# Good
from dataclasses import dataclass
from pathlib import Path

@dataclass
class QueryResult:
    query: str
    duration: float
    tool_calls: int

config_path = Path(__file__).parent / "config.json"
```

## Comments

- Comments should explain **why**, not **what** — the code itself should explain the "what".
- Use **section comments** with visual separators for major blocks.
- Keep comments **up to date** — stale comments are worse than no comments.

```python
# Good — explains the "why"
# MemorySaver provides in-memory checkpointing so the agent remembers
# previous messages within the same thread, enabling multi-turn conversations.
memory = MemorySaver()

# Avoid — restates the "what"
# Create a MemorySaver object
memory = MemorySaver()
```

## Testing Conventions

- Use **pytest** as the test framework.
- Name test files `test_<module>.py` and test functions `test_<behavior>()`.
- Use **fixtures** for shared setup.
- Follow the **Arrange-Act-Assert** pattern in test bodies.
- Test **edge cases** and error paths, not just the happy path.

## Security

- **Never** hardcode secrets, API keys, or passwords — always use environment variables or a secrets manager.
- Validate and sanitize **all external input** before processing.
- Use **restricted eval** or safer alternatives — never use bare `eval()` on untrusted input.
- Add **path traversal protection** when handling file operations.
- Set `__builtins__: {}` in eval namespaces to limit exposure.